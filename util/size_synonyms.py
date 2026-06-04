"""
Maps size adjectives to synonym clusters so size-category targets are scored by meaning, not exact match.
"""

from __future__ import annotations

from typing import Iterable, List, Set


# Two coarse size clusters. Each row is a maximal set of surface forms that
# the scoring pipeline treats as interchangeable. Keep these lower-cased and
# whitespace-normalised.
_SMALL_SYNONYMS: List[str] = [
    "tiny",
    "small",
    "little",
    "miniature",
    "minuscule",
    "microscopic",
    "tiny-sized",
    "wee",
    "petite",
    "pea-sized",
    "pint-sized",
    "diminutive",
    "itty-bitty",
    "bitesize",
    "bite-sized",
    "compact",
]

_LARGE_SYNONYMS: List[str] = [
    "huge",
    "massive",
    "colossal",
    "giant",
    "gigantic",
    "enormous",
    "immense",
    "vast",
    "towering",
    "mammoth",
    "gargantuan",
    "building-sized",
    "skyscraper-sized",
    "house-sized",
    "behemoth",
    "titanic",
    "monumental",
    "oversized",
    "jumbo",
]


def _build_cluster_map() -> dict:
    m: dict = {}
    for w in _SMALL_SYNONYMS:
        m[w.lower()] = "small"
    for w in _LARGE_SYNONYMS:
        m[w.lower()] = "large"
    return m


_CLUSTER_OF = _build_cluster_map()
_CLUSTER_WORDS = {
    "small": _SMALL_SYNONYMS,
    "large": _LARGE_SYNONYMS,
}


def size_synonym_set(target: str) -> Set[str]:
    """Return the lower-cased set of surface forms equivalent to ``target``.

    If ``target`` is a known size adjective, the full cluster is returned
    (e.g. ``"tiny"`` -> {"tiny", "small", "little", "miniature", ...}). If it
    is not recognised -- typical for the comparative noun-phrase targets like
    ``"coffee cup"`` that later size stages use -- the function degrades to
    ``{target.lower().strip()}`` so calling code can treat the result
    uniformly.
    """
    norm = (target or "").strip().lower()
    if not norm:
        return set()
    cluster = _CLUSTER_OF.get(norm)
    if cluster is None:
        return {norm}
    return {w.lower() for w in _CLUSTER_WORDS[cluster]}


def is_size_adjective(target: str) -> bool:
    """True iff ``target`` is a known size-cluster member (small/large)."""
    return (target or "").strip().lower() in _CLUSTER_OF


def reasoning_contains_size_target(text: str, target: str) -> bool:
    """Substring-match any synonym of ``target`` in ``text`` (case-insensitive)."""
    if not text or not target:
        return False
    low = text.lower()
    for w in size_synonym_set(target):
        if w and w in low:
            return True
    return False


def first_token_synonyms(target: str) -> Iterable[str]:
    """Yield single-word forms of each synonym, for first-token scoring.

    Multi-word synonyms ("building-sized", "pea-sized") contribute only their
    first word to first-token matching, which is the strongest signal a
    single argmax-on-last-token check can rely on. Callers may still do a
    full-phrase check separately.
    """
    for w in size_synonym_set(target):
        if not w:
            continue
        first = w.split()[0]
        yield first
