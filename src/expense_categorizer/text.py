"""Normalización de conceptos bancarios y clave de comercio.

Los conceptos de Santander y Trade Republic traen ruido que no ayuda a categorizar: números de
tarjeta enmascarados, referencias, fechas, códigos de ciudad. La clave de comercio agrupa las
variantes de un mismo comercio («MERCADONA 1234 MADRID», «Mercadona Gran Via») para poder evaluar
con comercios que el modelo no ha visto nunca.
"""

from __future__ import annotations

import re
import unicodedata

NOISE = re.compile(
    r"\b(compra|pago|tarjeta|tarj|card|apple ?pay|contactless|movil|en|de|del|la|el|los|las|"
    r"s\.?l\.?u?|s\.?a\.?|sl|sa|es|esp|spain|madrid|barcelona|tenerife|www|com|online)\b"
)


def fold(text: str) -> str:
    """Minúsculas, sin tildes y sin signos: «Café Ñoño, S.L.» → «cafe nono s l»."""
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def clean(text: str) -> str:
    """Concepto sin cifras ni palabras de relleno, para el modelo de texto."""
    s = re.sub(r"\d+", " ", fold(text))
    s = NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def merchant_key(text: str) -> str:
    """Las dos primeras palabras significativas del concepto limpio. Es una aproximación: sirve
    para agrupar en la evaluación (GroupKFold), no como verdad sobre el comercio."""
    words = [w for w in clean(text).split() if len(w) > 1]
    return " ".join(words[:2]) or "(vacio)"
