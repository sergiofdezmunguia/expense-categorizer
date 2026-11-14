"""Modelos comparados. Todos siguen la interfaz de scikit-learn: fit(df, y) / predict_proba(df).

- `Majority`: la clase más frecuente. Suelo de referencia.
- `MerchantMemory`: recuerda la categoría de cada comercio visto. Es lo que hacen las reglas:
  perfecto en lo conocido, nada en lo nuevo.
- `TextModel`: TF-IDF de caracteres y palabras del concepto + importe, signo, día de la semana y
  tipo de cuenta, con regresión logística. Puede generalizar a comercios nuevos («farmacia
  lopez» se parece a «farmacia garcia»).
- `Hybrid`: memoria si el comercio es conocido; si no, el modelo de texto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from threadpoolctl import threadpool_limits

from expense_categorizer.text import clean, merchant_key


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas derivadas que usan los modelos. Solo información del propio movimiento."""
    out = pd.DataFrame(index=df.index)
    out["text"] = df["description"].map(clean)
    out["merchant"] = df["description"].map(merchant_key)
    amount = df["amount_cents"].astype(float) / 100
    out["is_income"] = (amount > 0).astype(int)
    out["log_amount"] = np.log1p(amount.abs())
    out["weekday"] = pd.to_datetime(df["booking_date"]).dt.day_of_week.astype(str)
    out["account_kind"] = df.get(
        "account_kind", pd.Series("bank", index=df.index)
    ).astype(str)
    return out


class _Base(BaseEstimator, ClassifierMixin):
    def predict(self, df):
        proba = self.predict_proba(df)
        return self.classes_[proba.argmax(axis=1)]


class Majority(_Base):
    def fit(self, df, y):
        y = pd.Series(y)
        self.classes_ = np.array(sorted(y.unique()))
        freq = y.value_counts(normalize=True)
        self.prior_ = np.array([freq.get(c, 0.0) for c in self.classes_])
        return self

    def predict_proba(self, df):
        return np.tile(self.prior_, (len(df), 1))


class MerchantMemory(_Base):
    """Comercio conocido ⇒ su categoría más frecuente (probabilidad 1). Desconocido ⇒ el
    reparto global, con confianza baja: así la abstención lo descarta."""

    def __init__(self, min_purity: float = 0.9, min_count: int = 2):
        self.min_purity = min_purity
        self.min_count = min_count

    def fit(self, df, y):
        p = prepare(df)
        y = pd.Series(np.asarray(y), index=df.index)
        self.classes_ = np.array(sorted(y.unique()))
        # Solo comercios «puros»: si una clave mezcla categorías (transferencias, «sin concepto»),
        # recordarla empeora el resultado; mejor que decida el modelo de texto.
        stats = y.groupby(p["merchant"]).agg(
            top=lambda s: s.value_counts().index[0],
            purity=lambda s: s.value_counts(normalize=True).iloc[0],
            n="size",
        )
        keep = (stats["purity"] >= self.min_purity) & (stats["n"] >= self.min_count)
        self.memory_ = stats.loc[keep, "top"].to_dict()
        freq = y.value_counts(normalize=True)
        self.prior_ = np.array([freq.get(c, 0.0) for c in self.classes_])
        return self

    def known(self, df) -> np.ndarray:
        return prepare(df)["merchant"].isin(self.memory_.keys()).to_numpy()

    def predict_proba(self, df):
        merchants = prepare(df)["merchant"]
        out = np.tile(self.prior_, (len(df), 1))
        index = {c: i for i, c in enumerate(self.classes_)}
        for row, m in enumerate(merchants):
            if m in self.memory_:
                out[row] = 0.0
                out[row, index[self.memory_[m]]] = 1.0
        return out


def _text_pipeline(C: float) -> Pipeline:
    text = FeatureUnion(
        [
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
                ),
            ),
            (
                "words",
                TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True
                ),
            ),
        ]
    )
    features = ColumnTransformer(
        [
            ("text", text, "text"),
            ("num", FunctionTransformer(), ["is_income", "log_amount"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["weekday", "account_kind"],
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
        ]
    )


class TextModel(_Base):
    def __init__(self, C: float = 10.0):
        self.C = C

    # Un solo hilo de BLAS/OpenMP: en WSL, con todos los núcleos, un ajuste tarda 17 s en vez
    # de 0,4 s (sobresuscripción de hilos), con el mismo resultado.
    def fit(self, df, y):
        with threadpool_limits(1):
            self.pipe_ = _text_pipeline(self.C).fit(prepare(df), np.asarray(y))
        self.classes_ = self.pipe_.classes_
        return self

    def predict_proba(self, df):
        with threadpool_limits(1):
            return self.pipe_.predict_proba(prepare(df))


class Hybrid(_Base):
    def __init__(self, C: float = 10.0):
        self.C = C

    def fit(self, df, y):
        self.memory_ = MerchantMemory().fit(df, y)
        self.text_ = TextModel(C=self.C).fit(df, y)
        self.classes_ = self.text_.classes_
        return self

    def predict_proba(self, df):
        out = self.text_.predict_proba(df)
        known = self.memory_.known(df)
        if known.any():
            mem = self.memory_.predict_proba(df)
            cols = [list(self.memory_.classes_).index(c) if c in self.memory_.classes_ else None
                    for c in self.classes_]  # fmt: skip
            aligned = np.column_stack(
                [mem[:, j] if j is not None else np.zeros(len(df)) for j in cols]
            )
            out[known] = aligned[known]
        return out


MODELS = {
    "mayoritaria": Majority,
    "memoria de comercios (≈ reglas)": MerchantMemory,
    "texto (TF-IDF + logística)": TextModel,
    "híbrido (memoria + texto)": Hybrid,
}
