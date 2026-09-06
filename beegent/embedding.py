"""Text -> vector, for catalog matching. Optional: absent means no vectors, never an error."""

import array
import logging
import math
from typing import Callable

from beegent import config

_log = logging.getLogger(__name__)

#: One embedder per process. Loading the model costs seconds; matching costs microseconds.
_cached: dict = {}


def normalise(vec: list) -> list:
    """Unit length at WRITE time, so similarity later is a plain dot product."""
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def to_blob(vec: list) -> bytes:
    """float32 via stdlib `array` - numpy is fastembed's dependency, not ours."""
    return array.array("f", vec).tobytes()


def from_blob(blob: bytes) -> list:
    out = array.array("f")
    out.frombytes(blob)
    return out.tolist()


def dot(a: list, b: list) -> float:
    """Both sides are unit vectors, so this IS the cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


def band(scores: list) -> tuple | None:
    """The largest gap in a ranking IS the boundary the data implies - no labels needed."""
    ranked = sorted(scores, reverse=True)
    if len(ranked) < 2:
        return None
    _, i = max((ranked[i] - ranked[i + 1], i) for i in range(len(ranked) - 1))
    return ranked[i + 1], ranked[i]  # (highest rejected, lowest accepted)


def intersect(bands: list) -> tuple | None:
    """None means NO single threshold separates every query - never average that away."""
    real = [b for b in bands if b]
    if not real:
        return None
    low, high = max(b[0] for b in real), min(b[1] for b in real)
    return (low, high) if low < high else None


class FastEmbedder:
    """Local ONNX model. No API, no key, independent of LLM_BACKEND."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # imported late: the extra may not be installed

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        # Batched on purpose - one call for a whole run's links, or a whole backfill.
        return [normalise(list(v)) for v in self._model.embed(texts)]


def make_embedder(model_name: str | None = None) -> Callable | None:
    """None when embeddings are off or unavailable - callers write NULL vectors and carry on."""
    model_name = model_name if model_name is not None else config.EMBED_MODEL
    if not model_name:
        return None
    if model_name not in _cached:
        try:
            _cached[model_name] = FastEmbedder(model_name)
        except Exception as exc:  # missing extra, failed download, no disk
            _log.info(f"[embed] disabled ({type(exc).__name__}: {exc})")
            _cached[model_name] = None
    return _cached[model_name]
