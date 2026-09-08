"""Phase 1 tests: cost accounting, caching, offline replay, and the quota guard.

None of these call the network. The cache and guard are what keep the evaluation
reproducible and affordable, so they need to be tested without needing a key.
"""

from __future__ import annotations

import pytest

from kivi.config import Settings
from kivi.llm import pricing
from kivi.llm.cache import LLMCache, cache_key, pack_vector, unpack_vector
from kivi.llm.gemini import CacheMiss, GeminiClient, QuotaExceeded, _extract_text


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        GEMINI_API_KEY="test-key-not-real",
        KIVI_DB_PATH=str(tmp_path / "test.db"),
    )


@pytest.fixture()
def cache(settings: Settings) -> LLMCache:
    return LLMCache(settings.resolved_db_path, enabled=True)


# --- pricing ----------------------------------------------------------------


def test_known_model_prices_are_real():
    price = pricing.lookup("gemini-3.5-flash-lite")
    assert price is not None
    assert price.input_per_m == 0.30
    assert price.output_per_m == 2.50


def test_models_prefix_is_stripped():
    assert pricing.lookup("models/gemini-3.5-flash") == pricing.lookup("gemini-3.5-flash")


def test_versioned_suffix_falls_back_to_longest_prefix():
    # e.g. gemini-3.1-pro-preview-customtools
    assert pricing.lookup("gemini-3.1-pro-preview-customtools") is not None


def test_unknown_model_costs_none_not_zero():
    """Reporting $0.00 for a model we have no price for would be a lie."""
    assert pricing.lookup("some-future-model") is None
    assert pricing.estimate_cost("some-future-model", 1000, 1000) is None


def test_cost_arithmetic():
    # 1M in + 1M out on flash-lite = $0.30 + $2.50
    assert pricing.estimate_cost("gemini-3.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(2.80)


# --- cache keys -------------------------------------------------------------


def test_cache_key_is_stable_under_dict_reordering():
    a = cache_key("generate", "m", {"x": 1, "y": 2})
    b = cache_key("generate", "m", {"y": 2, "x": 1})
    assert a == b


def test_cache_key_separates_models_and_kinds():
    payload = {"x": 1}
    assert cache_key("generate", "a", payload) != cache_key("generate", "b", payload)
    assert cache_key("generate", "a", payload) != cache_key("embed", "a", payload)


# --- vector packing ---------------------------------------------------------


def test_vector_roundtrip_within_float32_precision():
    original = [0.1, -0.25, 1e-6, 0.9999]
    restored = unpack_vector(pack_vector(original))
    assert restored == pytest.approx(original, abs=1e-6)


# --- cache behaviour --------------------------------------------------------


def test_response_cache_roundtrip(cache: LLMCache):
    key = cache_key("generate", "m", {"p": "hello"})
    assert cache.get_response(key) is None

    cache.put_response(
        key, model="m", kind="generate", response={"ok": True},
        tokens_in=10, tokens_out=5, cost_usd=0.001,
    )
    hit = cache.get_response(key)
    assert hit is not None
    assert hit["response"] == {"ok": True}
    assert hit["tokens_in"] == 10
    assert hit["cost_usd"] == 0.001


def test_unknown_cost_survives_the_cache_as_null(cache: LLMCache):
    key = cache_key("generate", "m", {"p": "x"})
    cache.put_response(
        key, model="m", kind="generate", response={}, tokens_in=1, tokens_out=1,
        cost_usd=None,
    )
    assert cache.get_response(key)["cost_usd"] is None


def test_embedding_cache_is_keyed_by_model_and_dim(cache: LLMCache):
    cache.put_embedding("text", "model-a", 768, [0.5] * 768)
    assert cache.get_embedding("text", "model-a", 768) is not None
    assert cache.get_embedding("text", "model-b", 768) is None
    assert cache.get_embedding("text", "model-a", 1536) is None


def test_disabled_cache_never_serves(settings: Settings):
    disabled = LLMCache(settings.resolved_db_path, enabled=False)
    key = cache_key("generate", "m", {"p": "x"})
    disabled.put_response(
        key, model="m", kind="generate", response={"ok": 1}, tokens_in=1,
        tokens_out=1, cost_usd=0.0,
    )
    assert disabled.get_response(key) is None


# --- offline replay ---------------------------------------------------------


def test_offline_mode_raises_on_generation_cache_miss(tmp_path):
    settings = Settings(
        GEMINI_API_KEY="test-key-not-real",
        KIVI_DB_PATH=str(tmp_path / "off.db"),
        KIVI_OFFLINE=True,
    )
    with GeminiClient(settings) as client:
        with pytest.raises(CacheMiss):
            client.generate("never cached")


def test_offline_mode_serves_a_cached_generation(tmp_path):
    """A replay must return the cached answer without touching the network."""
    db = str(tmp_path / "off2.db")
    warm = Settings(GEMINI_API_KEY="k", KIVI_DB_PATH=db)
    client = GeminiClient(warm)

    payload = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"temperature": 0.0},
    }
    body = {"candidates": [{"content": {"parts": [{"text": "cached answer"}]}}]}
    client.cache.put_response(
        cache_key("generate", warm.gen_model, payload),
        model=warm.gen_model, kind="generate", response=body,
        tokens_in=3, tokens_out=2, cost_usd=0.0001,
    )

    offline = Settings(GEMINI_API_KEY="k", KIVI_DB_PATH=db, KIVI_OFFLINE=True)
    with GeminiClient(offline) as replay:
        result = replay.generate("hi")
    assert result.text == "cached answer"
    assert result.cached is True
    assert result.latency_ms == 0


def test_offline_mode_raises_on_embedding_cache_miss(tmp_path):
    settings = Settings(
        GEMINI_API_KEY="k", KIVI_DB_PATH=str(tmp_path / "off3.db"), KIVI_OFFLINE=True
    )
    with GeminiClient(settings) as client:
        with pytest.raises(CacheMiss):
            client.embed(["never cached"])


# --- quota guard ------------------------------------------------------------


def test_quota_guard_trips_at_the_configured_limit(tmp_path):
    settings = Settings(
        GEMINI_API_KEY="k",
        KIVI_DB_PATH=str(tmp_path / "q.db"),
        KIVI_DAILY_CALL_LIMIT=2,
    )
    client = GeminiClient(settings)
    client.cache.record_call(settings.gen_model, "generate")
    client.cache.record_call(settings.gen_model, "generate")

    with pytest.raises(QuotaExceeded):
        client.generate("this should never reach the network")


def test_quota_guard_is_off_when_limit_is_zero(tmp_path):
    settings = Settings(
        GEMINI_API_KEY="k",
        KIVI_DB_PATH=str(tmp_path / "q2.db"),
        KIVI_DAILY_CALL_LIMIT=0,
    )
    client = GeminiClient(settings)
    for _ in range(50):
        client.cache.record_call(settings.gen_model, "generate")
    client._check_quota(settings.gen_model)  # must not raise


def test_quota_is_counted_per_model(tmp_path):
    settings = Settings(
        GEMINI_API_KEY="k", KIVI_DB_PATH=str(tmp_path / "q3.db"),
        KIVI_DAILY_CALL_LIMIT=1,
    )
    client = GeminiClient(settings)
    client.cache.record_call("model-a", "generate")
    client._check_quota("model-b")  # a different model is unaffected
    with pytest.raises(QuotaExceeded):
        client._check_quota("model-a")


# --- response parsing -------------------------------------------------------


def test_extract_text_joins_multiple_parts():
    body = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}
    assert _extract_text(body) == "ab"


def test_extract_text_tolerates_a_blocked_or_empty_response():
    """A safety block returns no candidates; that is an empty answer, not a crash."""
    assert _extract_text({}) == ""
    assert _extract_text({"candidates": []}) == ""
    assert _extract_text({"candidates": [{"content": {}}]}) == ""


# --- config -----------------------------------------------------------------


def test_missing_key_is_reported_not_guessed(tmp_path):
    settings = Settings(GEMINI_API_KEY="", KIVI_DB_PATH=str(tmp_path / "n.db"))
    assert settings.has_key is False


def test_db_parent_directory_is_created(tmp_path):
    settings = Settings(
        GEMINI_API_KEY="k", KIVI_DB_PATH=str(tmp_path / "nested" / "deep" / "k.db")
    )
    assert settings.resolved_db_path.parent.is_dir()
