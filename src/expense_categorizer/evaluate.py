"""Evaluación honesta: dos divisiones y abstención con umbral elegido sin mirar la prueba.

- **Temporal:** entrena con lo anterior a la fecha de corte y prueba con lo posterior. Es el uso
  real: categorizar lo que llega.
- **Comercios nuevos (GroupKFold):** ningún comercio de la prueba aparece en el entrenamiento.
  Mide lo que las reglas no pueden hacer.

Además de accuracy y F1 macro, se mide la **cobertura con precisión objetivo**: qué parte de los
movimientos se puede categorizar sola con al menos un 90 % de aciertos, dejando el resto para
revisión. El umbral se elige con validación cruzada dentro del entrenamiento.

    uv run python -m expense_categorizer.evaluate --source synthetic
    uv run python -m expense_categorizer.evaluate --source personal-os   # datos reales, local
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, KFold

from expense_categorizer.models import MODELS
from expense_categorizer.text import merchant_key

TARGET_PRECISION = 0.90
MIN_PER_CLASS = 5


@dataclass
class Score:
    model: str
    split: str
    n_test: int
    accuracy: float
    macro_f1: float
    coverage_at_target: float
    precision_covered: float


def collapse_rare(
    y: pd.Series, min_count: int = MIN_PER_CLASS, other: str = "otros-gastos"
) -> pd.Series:
    """Categorías con menos de `min_count` ejemplos van a `other`: no se pueden aprender ni
    evaluar con un mínimo de fiabilidad."""
    counts = y.value_counts()
    rare = counts[counts < min_count].index
    return y.where(~y.isin(rare), other)


def pick_threshold(
    conf: np.ndarray, correct: np.ndarray, target: float = TARGET_PRECISION
) -> float:
    """Umbral de confianza más bajo cuya precisión en lo cubierto alcanza el objetivo."""
    order = np.argsort(-conf)
    hits = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(hits >= target)[0]
    if not len(ok):
        return 1.01  # nunca alcanza el objetivo: no cubre nada
    return float(conf[order][ok.max()])


def _inner_threshold(
    model, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None
) -> float:
    """La validación interna imita a la prueba: por comercio si la prueba es de comercios nuevos,
    aleatoria si es temporal (ahí la mayoría de comercios ya se han visto)."""
    conf, correct = np.zeros(len(X)), np.zeros(len(X), dtype=bool)
    if groups is None:
        splits = KFold(n_splits=3, shuffle=True, random_state=0).split(X)
    else:
        if groups.nunique() < 2:
            return 1.01
        splits = GroupKFold(n_splits=min(3, groups.nunique())).split(X, y, groups)
    for tr, va in splits:
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])
        conf[va] = p.max(axis=1)
        correct[va] = m.classes_[p.argmax(axis=1)] == y.iloc[va].to_numpy()
    return pick_threshold(conf, correct)


def score(name: str, model, X_tr, y_tr, X_te, y_te, split: str) -> Score:
    groups = (
        X_tr["description"].map(merchant_key) if split.startswith("comercios") else None
    )
    threshold = _inner_threshold(model, X_tr, y_tr, groups)
    m = clone(model).fit(X_tr, y_tr)
    p = m.predict_proba(X_te)
    pred = m.classes_[p.argmax(axis=1)]
    covered = p.max(axis=1) >= threshold
    y_true = y_te.to_numpy()
    return Score(
        model=name,
        split=split,
        n_test=len(y_te),
        accuracy=accuracy_score(y_true, pred),
        macro_f1=f1_score(
            y_true, pred, average="macro", labels=np.unique(y_true), zero_division=0
        ),
        coverage_at_target=float(covered.mean()),
        precision_covered=float((pred[covered] == y_true[covered]).mean())
        if covered.any()
        else float("nan"),
    )


def _pool(name: str, parts: list[Score]) -> Score:
    """Agrega los pliegues ponderando por movimientos; la precisión cubierta, por cubiertos."""
    n = sum(p.n_test for p in parts)
    avg = lambda f: sum(getattr(p, f) * p.n_test for p in parts) / n
    covered = [(p.coverage_at_target * p.n_test, p.precision_covered) for p in parts]
    total = sum(c for c, _ in covered)
    precision = sum(c * pr for c, pr in covered if c) / total if total else float("nan")
    return Score(name, "comercios nuevos", n, avg("accuracy"), avg("macro_f1"),
                 avg("coverage_at_target"), precision)  # fmt: skip


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("booking_date", kind="stable")
    cut = df["booking_date"].iloc[int(len(df) * (1 - test_frac))]
    return df[df["booking_date"] < cut], df[df["booking_date"] >= cut]


def run(df: pd.DataFrame, folds: int = 5) -> list[Score]:
    df = df.dropna(subset=["category"]).reset_index(drop=True)
    df["category"] = collapse_rare(df["category"])
    out: list[Score] = []
    tr, te = temporal_split(df)
    for name, cls in MODELS.items():
        out.append(
            score(name, cls(), tr, tr["category"], te, te["category"], "temporal")
        )
    groups = df["description"].map(merchant_key)
    for name, cls in MODELS.items():
        parts = []
        for i, (a, b) in enumerate(
            GroupKFold(n_splits=folds).split(df, df["category"], groups)
        ):
            parts.append(score(name, cls(), df.iloc[a], df["category"].iloc[a], df.iloc[b],
                               df["category"].iloc[b], f"comercios nuevos #{i}"))  # fmt: skip
        out.append(_pool(name, parts))
    return out


def report(scores: list[Score], title: str, df: pd.DataFrame) -> str:
    rows = [s for s in scores if "#" not in s.split]
    lines = [
        f"# {title}",
        "",
        (
            f"{len(df)} movimientos etiquetados, {df['category'].nunique()} categorías, "
            f"{df['description'].map(merchant_key).nunique()} comercios (clave aproximada)."
        ),
        "",
        (
            f"Cobertura = parte que se categoriza sola con un umbral elegido para ≥ {TARGET_PRECISION:.0%} "
            "de precisión en validación interna; «precisión cubierta» es la que se obtiene de verdad en la prueba."
        ),
        "",
        "| División | Modelo | Accuracy | F1 macro | Cobertura | Precisión cubierta |",
        "|---|---|---|---|---|---|",
    ]
    for s in rows:
        pc = (
            "—"
            if s.precision_covered != s.precision_covered
            else f"{s.precision_covered:.0%}"
        )
        lines.append(f"| {s.split} | {s.model} | {s.accuracy:.0%} | {s.macro_f1:.2f} | "
                     f"{s.coverage_at_target:.0%} | {pc} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", choices=["synthetic", "personal-os"], default="synthetic"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.source == "synthetic":
        from expense_categorizer.synthetic import generate

        df, title, out = (
            generate(),
            "Resultados con datos sintéticos",
            Path("results/synthetic.md"),
        )
    else:
        from expense_categorizer.data import load_personal_os

        df, title, out = (
            load_personal_os(),
            "Resultados con mis datos reales",
            Path("results/real/report.md"),
        )
    out = args.out or out
    scores = run(df)
    text = report(scores, title, df.dropna(subset=["category"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
