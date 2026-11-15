"""LLM local en modo zero-shot, como referencia (no como propuesta de producción).

Qwen2.5-1.5B-Instruct (Apache 2.0) con `transformers`, en local y en GPU si la hay. No hace
falta Ollama ni ninguna API: los pesos se descargan una vez de Hugging Face.

En vez de generar texto libre y adivinar a qué categoría se refiere, **puntúa cada categoría**:
la probabilidad que el modelo da a escribir exactamente ese nombre (más fin de turno) como
respuesta. Normalizadas entre las categorías, son las probabilidades de `predict_proba`, así que
la abstención funciona igual que con los demás modelos. Del entrenamiento solo usa la lista de
categorías: el umbral se elige con validación interna, como siempre.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from expense_categorizer import embeddings
from expense_categorizer.models import _Base, prepare

QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = (
    "Eres un clasificador de movimientos bancarios de una persona en España. "
    "Responde solo con el nombre exacto de una categoría de la lista."
)

_models: dict[tuple[str, str], tuple] = {}
_cache: dict[tuple[str, str, str], float] = {}


def available() -> bool:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _load(name: str):
    dev = embeddings.device()
    if (name, dev) not in _models:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(embeddings.TORCH_THREADS)
        tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if dev == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
        _models[(name, dev)] = (tok, model, dev)
    return _models[(name, dev)]


def _prompt(tok, labels, text: str, amount: float, kind: str) -> str:
    user = (
        f"Categorías: {', '.join(labels)}\n"
        f"Movimiento: «{text or 'sin concepto'}», importe {amount:+.2f} €, cuenta {kind}.\n"
        "¿Qué categoría es?"
    )
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _score_labels(name: str, prompt: str, labels: list[str]) -> np.ndarray:
    """log P(etiqueta + fin | prompt) para cada etiqueta: el prompt se procesa una vez y su caché
    de claves/valores se reutiliza para todas las etiquetas en un solo lote."""
    import torch

    tok, model, dev = _load(name)
    seqs = [
        tok(lab, add_special_tokens=False).input_ids + [tok.eos_token_id]
        for lab in labels
    ]
    width = max(map(len, seqs))
    lab = torch.full((len(seqs), width), tok.eos_token_id, device=dev)
    mask = torch.zeros((len(seqs), width), dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        lab[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        first = torch.log_softmax(out.logits[0, -1].float(), -1)
        past = out.past_key_values
        past.batch_repeat_interleave(len(seqs))
        attn = torch.cat(
            [torch.ones((len(seqs), ids.shape[1]), dtype=torch.long, device=dev), mask],
            1,
        )
        rest = torch.log_softmax(
            model(lab, past_key_values=past, attention_mask=attn).logits.float(), -1
        )
    lp = first[lab[:, 0]].clone()
    if width > 1:
        nxt = rest[:, :-1].gather(2, lab[:, 1:, None])[..., 0]
        lp += (nxt * mask[:, 1:]).sum(1)
    return lp.cpu().numpy()


class ZeroShotLLM(_Base):
    def __init__(self, model_name: str = QWEN):
        self.model_name = model_name

    def fit(self, df, y):
        self.classes_ = np.array(sorted(pd.Series(y).unique()))
        return self

    def frozen_bytes(self) -> int:
        _, model, _ = _load(self.model_name)
        return sum(p.numel() * p.element_size() for p in model.parameters())

    def predict_proba(self, df):
        tok, _, _ = _load(self.model_name)
        labels = list(self.classes_)
        p = prepare(df)
        amounts = df["amount_cents"].astype(float).to_numpy() / 100
        out = np.zeros((len(df), len(labels)))
        for row, (text, amount, kind) in enumerate(
            zip(p["text"], amounts, p["account_kind"])
        ):
            prompt = _prompt(tok, labels, text, amount, kind)
            keys = [(self.model_name, prompt, lab) for lab in labels]
            if embeddings.caching() and all(k in _cache for k in keys):
                lp = np.array([_cache[k] for k in keys])
            else:
                lp = _score_labels(self.model_name, prompt, labels)
                if embeddings.caching():
                    _cache.update(zip(keys, lp.tolist()))
            z = np.exp(lp - lp.max())
            out[row] = z / z.sum()
        return out
