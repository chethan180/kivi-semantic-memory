"""Gemini client over the REST API.

Deliberately `httpx` rather than the `google-genai` SDK: retries, caching, cost
accounting and the offline replay mode all need to sit *inside* the call path,
and the review agent has one less dependency to install.

Every call returns an `LLMResult` carrying tokens, cost and latency, because the
evaluation has to report those and they are impossible to reconstruct afterwards.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from kivi.config import Settings, get_settings
from kivi.llm import pricing
from kivi.llm.cache import LLMCache, cache_key

log = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Transient conditions worth retrying. 429 is rate limiting, 5xx is Google's side.
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """A model call failed in a way retrying will not fix."""


class QuotaExceeded(LLMError):
    """The configured daily call guard tripped before the call was made."""


class CacheMiss(LLMError):
    """Offline replay mode was on and the call was not in the cache."""


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    latency_ms: int = 0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse the response as JSON. Only valid for structured-output calls."""
        text = self.text.strip()
        # Tolerate a fenced block even though responseMimeType should prevent one.
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return valid JSON: {exc}") from exc


class GeminiClient:
    def __init__(
        self,
        settings: Settings | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or LLMCache(
            self.settings.resolved_db_path, enabled=self.settings.llm_cache
        )
        self._client: httpx.Client | None = None

    # --- transport -----------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.settings.request_timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> GeminiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_key(self) -> str:
        if not self.settings.has_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.settings.gemini_api_key

    def _check_quota(self, model: str) -> None:
        limit = self.settings.daily_call_limit
        if limit and self.cache.calls_today(model) >= limit:
            raise QuotaExceeded(
                f"daily call guard reached for {model}"
                f" ({limit} calls). Raise or disable KIVI_DAILY_CALL_LIMIT."
            )

    def _post(self, path: str, payload: dict[str, Any], model: str, kind: str) -> dict[str, Any]:
        """POST with bounded exponential backoff and jitter."""
        key = self._require_key()
        self._check_quota(model)
        url = f"{API_ROOT}/{path}"
        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries):
            try:
                response = self.client.post(
                    url, params={"key": key}, json=payload
                )
            except httpx.RequestError as exc:
                last_error = exc
            else:
                self.cache.record_call(model, kind)
                if response.status_code == 200:
                    return response.json()
                if response.status_code not in RETRY_STATUS:
                    raise LLMError(
                        f"{response.status_code} from {path}: {response.text[:500]}"
                    )
                last_error = LLMError(
                    f"{response.status_code} from {path}: {response.text[:200]}"
                )

            if attempt < self.settings.max_retries - 1:
                delay = min(2**attempt, 16) + random.uniform(0, 0.5)
                log.warning(
                    "retrying %s in %.1fs (attempt %d/%d): %s",
                    path, delay, attempt + 1, self.settings.max_retries, last_error,
                )
                time.sleep(delay)

        raise LLMError(f"{path} failed after {self.settings.max_retries} attempts: {last_error}")

    # --- generation ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        """One-shot generation. Pass `schema` for JSON-mode structured output."""
        model = model or self.settings.gen_model

        generation_config: dict[str, Any] = {"temperature": temperature}
        if max_output_tokens:
            generation_config["maxOutputTokens"] = max_output_tokens
        if schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = schema

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        key = cache_key("generate", model, payload)
        hit = self.cache.get_response(key)
        if hit is not None:
            return LLMResult(
                text=_extract_text(hit["response"]),
                model=hit["model"],
                tokens_in=hit["tokens_in"],
                tokens_out=hit["tokens_out"],
                cost_usd=hit["cost_usd"],
                latency_ms=0,
                cached=True,
                raw=hit["response"],
            )

        if self.settings.offline:
            raise CacheMiss(
                "KIVI_OFFLINE=true and this generation is not cached. "
                "Run once with KIVI_OFFLINE=false to populate the cache."
            )

        started = time.perf_counter()
        body = self._post(f"models/{model}:generateContent", payload, model, "generate")
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = body.get("usageMetadata", {})
        tokens_in = int(usage.get("promptTokenCount", 0))
        # Thinking tokens are billed as output; include them or cost under-reports.
        tokens_out = int(usage.get("candidatesTokenCount", 0)) + int(
            usage.get("thoughtsTokenCount", 0)
        )
        cost = pricing.estimate_cost(model, tokens_in, tokens_out)

        self.cache.put_response(
            key,
            model=model,
            kind="generate",
            response=body,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )

        return LLMResult(
            text=_extract_text(body),
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_ms=latency_ms,
            cached=False,
            raw=body,
        )

    def converse(
        self,
        contents: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_mode: str = "AUTO",
        allowed_tools: list[str] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = 4096,
    ) -> tuple[dict[str, Any], LLMResult]:
        """Multi-turn generation with function calling.

        Returns the raw candidate alongside the usual result, because the caller
        needs the `functionCall` parts, not just the text. `contents` is the full
        conversation so far, so the cache key covers the whole exchange - the
        same conversation replays for free and, in offline mode, provably without
        new inference.
        """
        model = model or self.settings.gen_model_heavy

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                **({"maxOutputTokens": max_output_tokens} if max_output_tokens else {}),
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]
            # AUTO lets the model decide when it has enough; the caller's loop
            # bounds the rest. NONE forces prose.
            #
            # NONE has to be set explicitly, and the declarations have to stay:
            # simply omitting `tools` does NOT stop the model calling them once
            # the conversation already contains functionCall/functionResponse
            # parts. It keeps the pattern going and returns another call with no
            # text at all, so the turn ends with nothing to show the person.
            config: dict[str, Any] = {"mode": tool_mode}
            if allowed_tools:
                # ANY + allowedFunctionNames makes the choice for the model
                # instead of asking it to make one. Used where the request shape
                # is unambiguous in the text itself, so spending a round trip on
                # a decision we can already make would be paying for nothing.
                config = {
                    "mode": "ANY", "allowedFunctionNames": list(allowed_tools)
                }
            payload["toolConfig"] = {"functionCallingConfig": config}

        key = cache_key("converse", model, payload)
        hit = self.cache.get_response(key)
        if hit is not None:
            body = hit["response"]
            return _first_candidate(body), LLMResult(
                text=_extract_text(body), model=hit["model"],
                tokens_in=hit["tokens_in"], tokens_out=hit["tokens_out"],
                cost_usd=hit["cost_usd"], latency_ms=0, cached=True, raw=body,
            )

        if self.settings.offline:
            raise CacheMiss(
                "KIVI_OFFLINE=true and this conversation turn is not cached."
            )

        started = time.perf_counter()
        body = self._post(f"models/{model}:generateContent", payload, model, "converse")
        latency_ms = int((time.perf_counter() - started) * 1000)

        usage = body.get("usageMetadata", {})
        tokens_in = int(usage.get("promptTokenCount", 0))
        tokens_out = int(usage.get("candidatesTokenCount", 0)) + int(
            usage.get("thoughtsTokenCount", 0)
        )
        cost = pricing.estimate_cost(model, tokens_in, tokens_out)

        self.cache.put_response(
            key, model=model, kind="converse", response=body,
            tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost,
        )
        return _first_candidate(body), LLMResult(
            text=_extract_text(body), model=model, tokens_in=tokens_in,
            tokens_out=tokens_out, cost_usd=cost, latency_ms=latency_ms,
            cached=False, raw=body,
        )

    # --- embeddings ----------------------------------------------------------

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dim: int | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Embed texts, serving whatever is already cached and batching the rest.

        `task_type` should be RETRIEVAL_DOCUMENT when storing and RETRIEVAL_QUERY
        when searching; Gemini embeds those asymmetrically and mixing them costs
        real retrieval quality.
        """
        model = model or self.settings.embed_model
        dim = dim or self.settings.embed_dim

        results: list[list[float] | None] = [None] * len(texts)
        pending: list[int] = []

        for i, text in enumerate(texts):
            cached = self.cache.get_embedding(_embed_cache_text(text, task_type), model, dim)
            if cached is not None:
                results[i] = cached
            else:
                pending.append(i)

        if pending and self.settings.offline:
            raise CacheMiss(
                f"KIVI_OFFLINE=true and {len(pending)} embedding(s) are not cached."
            )

        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": texts[i]}]},
                        "taskType": task_type,
                        "outputDimensionality": dim,
                    }
                    for i in chunk
                ]
            }
            body = self._post(
                f"models/{model}:batchEmbedContents", payload, model, "embed"
            )
            embeddings = body.get("embeddings", [])
            if len(embeddings) != len(chunk):
                raise LLMError(
                    f"expected {len(chunk)} embeddings, got {len(embeddings)}"
                )
            for idx, item in zip(chunk, embeddings):
                vector = [float(v) for v in item.get("values", [])]
                if not vector:
                    raise LLMError("embedding response contained no values")
                results[idx] = vector
                self.cache.put_embedding(
                    _embed_cache_text(texts[idx], task_type), model, dim, vector
                )

        missing = [i for i, v in enumerate(results) if v is None]
        if missing:
            raise LLMError(f"embeddings missing for indices {missing}")
        return [v for v in results if v is not None]

    def embed_one(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed([text], **kwargs)[0]

    # --- introspection -------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        """Live model list. Used by `kivi doctor` to prove the key works."""
        key = self._require_key()
        response = self.client.get(f"{API_ROOT}/models", params={"key": key})
        if response.status_code != 200:
            raise LLMError(
                f"{response.status_code} listing models: {response.text[:300]}"
            )
        return response.json().get("models", [])


def _first_candidate(body: dict[str, Any]) -> dict[str, Any]:
    candidates = body.get("candidates") or []
    return candidates[0] if candidates else {}


def function_calls(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract functionCall parts from a candidate, in order."""
    parts = (candidate.get("content") or {}).get("parts") or []
    return [p["functionCall"] for p in parts if "functionCall" in p]


def _embed_cache_text(text: str, task_type: str) -> str:
    """Task type changes the vector, so it must be part of the cache identity."""
    return f"{task_type}\x00{text}"


def _extract_text(body: dict[str, Any]) -> str:
    """Pull text out of a generateContent response.

    A response can legitimately have no text part - a safety block, or a
    thinking-only candidate that hit the token ceiling. Return empty rather than
    raising so the caller can decide what an empty answer means.
    """
    candidates = body.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)
