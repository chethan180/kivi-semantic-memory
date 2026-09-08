"""Development corpus generation.

The assignment asks for ~500 transcript-like records and an evaluation that can
be scored. That only works if we know the truth in advance, so the split here is
strict: **the plan is deterministic Python, the LLM only writes the surface
text.** A model that also decided what was true would make the ground truth a
restatement of its own output, and the evaluation meaningless.
"""

from kivi.corpus.profiles import PROFILES, ContextProfile

__all__ = ["PROFILES", "ContextProfile"]
