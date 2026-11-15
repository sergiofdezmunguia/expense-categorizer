"""Modelos comparados. Todos siguen la interfaz de scikit-learn: fit(df, y) / predict_proba(df).

- `Majority`: la clase más frecuente. Suelo de referencia.
- `MerchantMemory`: recuerda la categoría de cada comercio visto. Es lo que hacen las reglas:
  perfecto en lo conocido, nada en lo nuevo.
- `TextModel`: TF-IDF de caracteres y palabras del concepto + importe, signo, día de la semana y
  tipo de cuenta, con regresión logística. Puede generalizar a comercios nuevos («farmacia
  lopez» se parece a «farmacia garcia»).
- `Hybrid`: memoria si el comercio es conocido; si no, el modelo de texto.
- `EmbeddingModel` (v2): lo mismo que `TextModel`, pero el texto entra como embedding de frase
  multilingüe (`embeddings.py`) en vez de TF-IDF, o como ambos concatenados.
- `ZeroShotEmbedding` (v2): referencia sin entrenar el clasificador: similitud coseno entre el
  concepto y el nombre de cada categoría.
- `TunedC` (v2): envuelve cualquiera de los anteriores con logística y elige su C por validación
  interna agrupada por comercio.
- `llm.ZeroShotLLM` (v2, opcional): LLM local que puntúa el nombre de cada categoría.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
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


def _tfidf() -> FeatureUnion:
    return FeatureUnion(
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


def _text_pipeline(C: float, text="tfidf", embedder: str | None = None) -> Pipeline:
    if text == "tfidf":
        text_features = _tfidf()
    else:
        from expense_categorizer.embeddings import SentenceEmbedder

        emb = SentenceEmbedder(embedder)
        text_features = (
            emb if text == "emb" else FeatureUnion([("tfidf", _tfidf()), ("emb", emb)])
        )
    features = ColumnTransformer(
        [
            ("text", text_features, "text"),
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


class EmbeddingModel(TextModel):
    """`text="emb"`: embedding + importe/signo/día/cuenta. `text="tfidf+emb"`: ambos bloques de
    texto concatenados (los dos con norma 1 por fila, así pesan parecido)."""

    def __init__(
        self, C: float = 10.0, model_name: str | None = None, text: str = "emb"
    ):
        self.C = C
        self.model_name = model_name
        self.text = text

    def fit(self, df, y):
        from expense_categorizer.embeddings import E5_SMALL

        with threadpool_limits(1):
            pipe = _text_pipeline(self.C, self.text, self.model_name or E5_SMALL)
            self.pipe_ = pipe.fit(prepare(df), np.asarray(y))
        self.classes_ = self.pipe_.classes_
        return self

    def frozen_bytes(self) -> int:
        from expense_categorizer.embeddings import E5_SMALL, encoder_bytes

        return encoder_bytes(self.model_name or E5_SMALL)


class ZeroShotEmbedding(_Base):
    """Sin aprender de los ejemplos: cada movimiento va a la categoría cuyo nombre («farmacia»,
    «comida a domicilio») tiene el embedding más parecido. Solo usa del entrenamiento la lista de
    categorías. Las probabilidades son un softmax de las similitudes con temperatura fija."""

    def __init__(self, model_name: str | None = None, temperature: float = 0.02):
        self.model_name = model_name
        self.temperature = temperature

    def fit(self, df, y):
        self.classes_ = np.array(sorted(pd.Series(y).unique()))
        return self

    def frozen_bytes(self) -> int:
        from expense_categorizer.embeddings import E5_SMALL, encoder_bytes

        return encoder_bytes(self.model_name or E5_SMALL)

    def predict_proba(self, df):
        from expense_categorizer.embeddings import E5_SMALL, encode

        name = self.model_name or E5_SMALL
        labels = [c.replace("-", " ") for c in self.classes_]
        sims = encode(name, prepare(df)["text"]) @ encode(name, labels).T
        z = np.exp((sims - sims.max(axis=1, keepdims=True)) / self.temperature)
        return z / z.sum(axis=1, keepdims=True)


class TunedC(_Base):
    """Elige la regularización C de `base` con validación cruzada **dentro del entrenamiento**,
    agrupada por comercio (lo que se quiere es generalizar a comercios nuevos), y reajusta con
    todo el entrenamiento. Con C fijo a 10, los embeddings infraajustaban (90 % de aciertos en
    el propio entrenamiento real, frente al 100 % de TF-IDF): su escala es muy distinta."""

    def __init__(self, base=None, grid=(1.0, 10.0, 100.0, 1000.0), folds: int = 3):
        self.base = base
        self.grid = grid
        self.folds = folds

    def fit(self, df, y):
        base = self.base if self.base is not None else TextModel()
        y = np.asarray(y)
        groups = df["description"].map(merchant_key)
        self.C_ = base.C
        if groups.nunique() >= self.folds:
            splits = list(GroupKFold(n_splits=self.folds).split(df, y, groups))
            acc = []
            for c in self.grid:
                hits = 0
                for tr, va in splits:
                    m = clone(base).set_params(C=c).fit(df.iloc[tr], y[tr])
                    hits += (m.predict(df.iloc[va]) == y[va]).sum()
                acc.append(hits / len(y))
            self.C_ = self.grid[int(np.argmax(acc))]  # empate → la C más pequeña
        self.model_ = clone(base).set_params(C=self.C_).fit(df, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, df):
        return self.model_.predict_proba(df)

    def frozen_bytes(self) -> int:
        m = self.model_
        return m.frozen_bytes() if hasattr(m, "frozen_bytes") else 0


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
BASELINE = "texto (TF-IDF + logística)"


def embedding_models() -> dict:
    """Modelos v2, declarados antes de ver resultados: dos codificadores de ~118 M parámetros y
    la combinación con TF-IDF solo con e5 (elegido a priori, no por su resultado). Todos con C
    elegido por validación interna (`TunedC`), y TF-IDF igual, como control justo."""
    from expense_categorizer.embeddings import E5_SMALL, MINILM

    def tuned(**kw):
        return lambda: TunedC(EmbeddingModel(**kw))

    return {
        "zero-shot (similitud e5 con el nombre)": lambda: ZeroShotEmbedding(E5_SMALL),
        "TF-IDF + logística, C interno": lambda: TunedC(TextModel()),
        "e5-small + logística, C interno": tuned(model_name=E5_SMALL),
        "MiniLM-L12 + logística, C interno": tuned(model_name=MINILM),
        "TF-IDF + e5-small + logística, C interno": tuned(
            model_name=E5_SMALL, text="tfidf+emb"
        ),
    }


def fixed_c_embedding_models() -> dict:
    """Primer intento v2 (C = 10 fijo, como TF-IDF). Se conserva para reproducir el informe."""
    from expense_categorizer.embeddings import E5_SMALL, MINILM

    return {
        "e5-small + logística, C=10": lambda: EmbeddingModel(model_name=E5_SMALL),
        "MiniLM-L12 + logística, C=10": lambda: EmbeddingModel(model_name=MINILM),
        "TF-IDF + e5-small + logística, C=10": lambda: EmbeddingModel(
            model_name=E5_SMALL, text="tfidf+emb"
        ),
    }


def llm_models() -> dict:
    from expense_categorizer.llm import QWEN, ZeroShotLLM

    return {"LLM zero-shot (Qwen2.5-1.5B)": lambda: ZeroShotLLM(QWEN)}
