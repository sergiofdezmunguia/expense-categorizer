"""Embeddings multilingües pequeños, locales, con caché.

Un embedding de frase convierte «farmacia zurbaran» en un vector de 384 números que queda cerca
de «parafarmacia del sol» aunque no compartan letras. Es lo que TF-IDF no sabe hacer: TF-IDF solo
generaliza si el comercio nuevo comparte trozos de palabra con los vistos.

- El modelo se descarga una vez de Hugging Face (`~/.cache/huggingface`) y corre en local, en GPU
  si la hay. Ningún texto sale del PC.
- Caché en memoria por (modelo, texto): la validación cruzada codifica los mismos conceptos
  decenas de veces. No se guarda en disco a propósito: con datos reales, los vectores son
  derivados de los conceptos y no deben quedar sueltos en ningún sitio.
- `no_cache()` la desactiva para medir la latencia de verdad.

Requiere el extra `embeddings` (`uv sync --extra embeddings`).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

E5_SMALL = "intfloat/multilingual-e5-small"
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Prefijo que exige la familia e5; las demás no lo usan.
_PREFIX = {E5_SMALL: "query: "}
# Pocos hilos: en WSL, con todos los núcleos, BLAS/OpenMP se pisan (ver TextModel).
TORCH_THREADS = int(os.environ.get("EC_TORCH_THREADS", "1"))

_encoders: dict[str, object] = {}
_cache: dict[tuple[str, str], np.ndarray] = {}
_use_cache = True


def available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _encoder(name: str):
    if name not in _encoders:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(TORCH_THREADS)
        _encoders[name] = SentenceTransformer(name, device=device())
    return _encoders[name]


def encoder_bytes(name: str) -> int:
    """Tamaño en memoria de los pesos del codificador."""
    model = _encoder(name)
    return sum(p.numel() * p.element_size() for p in model.parameters())


@contextmanager
def no_cache():
    global _use_cache
    old, _use_cache = _use_cache, False
    try:
        yield
    finally:
        _use_cache = old


def caching() -> bool:
    return _use_cache


def clear_cache() -> None:
    _cache.clear()


def encode(name: str, texts) -> np.ndarray:
    texts = [str(t) for t in texts]
    todo = sorted({t for t in texts if not _use_cache or (name, t) not in _cache})
    vectors: dict[str, np.ndarray] = {}
    if todo:
        prefix = _PREFIX.get(name, "")
        out = _encoder(name).encode(
            [prefix + (t or "(vacio)") for t in todo],
            batch_size=128,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = dict(zip(todo, out.astype(np.float32)))
        if _use_cache:
            _cache.update({(name, t): v for t, v in vectors.items()})
    return np.vstack([vectors[t] if t in vectors else _cache[(name, t)] for t in texts])


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Transformador de scikit-learn: columna de texto → embeddings normalizados. No aprende
    nada de los datos (el codificador está congelado), así que no puede filtrar la prueba."""

    def __init__(self, model_name: str = E5_SMALL):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return encode(self.model_name, list(X))
