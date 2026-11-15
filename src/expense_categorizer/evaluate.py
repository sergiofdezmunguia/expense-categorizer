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
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold

from expense_categorizer import embeddings
from expense_categorizer.models import (
    BASELINE,
    MODELS,
    embedding_models,
    fixed_c_embedding_models,
    llm_models,
)
from expense_categorizer.text import merchant_key

TARGET_PRECISION = 0.90
MIN_PER_CLASS = 5


@dataclass
class Preds:
    """Predicciones fila a fila en la prueba; permiten agregar pliegues y hacer bootstrap."""

    y_true: np.ndarray
    pred: np.ndarray
    conf: np.ndarray
    covered: np.ndarray
    groups: np.ndarray  # clave de comercio: unidad del bootstrap

    @staticmethod
    def concat(parts: list[Preds]) -> Preds:
        return Preds(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                       ("y_true", "pred", "conf", "covered", "groups")))  # fmt: skip


@dataclass
class Score:
    model: str
    split: str
    n_test: int
    accuracy: float
    macro_f1: float
    coverage_at_target: float
    precision_covered: float
    preds: Preds | None = field(default=None, repr=False, compare=False)
    ci: dict = field(default_factory=dict, repr=False, compare=False)
    vs_baseline: tuple | None = field(default=None, repr=False, compare=False)


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


def metrics(name: str, split: str, p: Preds) -> Score:
    hit = p.pred == p.y_true
    return Score(
        model=name,
        split=split,
        n_test=len(hit),
        accuracy=float(hit.mean()),
        macro_f1=f1_score(
            p.y_true,
            p.pred,
            average="macro",
            labels=np.unique(p.y_true),
            zero_division=0,
        ),
        coverage_at_target=float(p.covered.mean()),
        precision_covered=float(hit[p.covered].mean())
        if p.covered.any()
        else float("nan"),
        preds=p,
    )


def predict_split(model, X_tr, y_tr, X_te, y_te, split: str) -> Preds:
    groups = (
        X_tr["description"].map(merchant_key) if split.startswith("comercios") else None
    )
    threshold = _inner_threshold(model, X_tr, y_tr, groups)
    m = clone(model).fit(X_tr, y_tr)
    p = m.predict_proba(X_te)
    conf = p.max(axis=1)
    return Preds(
        y_true=y_te.to_numpy(),
        pred=m.classes_[p.argmax(axis=1)],
        conf=conf,
        covered=conf >= threshold,
        groups=X_te["description"].map(merchant_key).to_numpy(),
    )


def score(name: str, model, X_tr, y_tr, X_te, y_te, split: str) -> Score:
    return metrics(name, split, predict_split(model, X_tr, y_tr, X_te, y_te, split))


def _cluster_resamples(groups: np.ndarray, n: int, seed: int = 0):
    """Bootstrap por comercio: los movimientos de un mismo comercio no son independientes, así
    que se remuestrean comercios enteros. Da intervalos más anchos (y más honestos)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    rows = [np.where(codes == g)[0] for g in range(len(uniq))]
    for _ in range(n):
        pick = rng.integers(len(rows), size=len(rows))
        yield np.concatenate([rows[g] for g in pick])


def bootstrap(p: Preds, n: int = 1000, seed: int = 0) -> dict:
    """IC 95 % (percentiles) de accuracy, cobertura y precisión cubierta. El umbral queda fijo:
    se remuestrea la prueba, no se vuelve a elegir nada."""
    hit = p.pred == p.y_true
    acc, cov, prec = [], [], []
    for idx in _cluster_resamples(p.groups, n, seed):
        acc.append(hit[idx].mean())
        c = p.covered[idx]
        cov.append(c.mean())
        if c.any():
            prec.append(hit[idx][c].mean())
    q = lambda v: (
        (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else None
    )
    return {"accuracy": q(acc), "coverage": q(cov), "precision_covered": q(prec)}


def paired_diff(p: Preds, base: Preds, n: int = 1000, seed: int = 0) -> tuple:
    """Diferencia de accuracy frente a la referencia sobre los mismos remuestreos (mismas filas,
    mismo orden): (diferencia, IC 2,5 %, IC 97,5 %)."""
    assert np.array_equal(p.y_true, base.y_true) and np.array_equal(
        p.groups, base.groups
    )
    d = (p.pred == p.y_true).astype(float) - (base.pred == base.y_true)
    boots = [d[idx].mean() for idx in _cluster_resamples(p.groups, n, seed)]
    return (
        float(d.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


RC_LEVELS = (0.25, 0.5, 0.75)
CPU_COST_ROWS = 100


def risk_coverage(p: Preds) -> dict:
    """Curva riesgo-cobertura descriptiva: precisión si se cubre el x % más seguro, y AURC (error
    medio a lo largo de toda la curva; menor es mejor). No elige ningún umbral: solo describe cómo
    ordena la confianza de cada modelo."""
    hit = (p.pred == p.y_true)[np.argsort(-p.conf, kind="stable")]
    prec = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    out = {
        lvl: float(prec[max(0, int(np.ceil(lvl * len(hit))) - 1)]) for lvl in RC_LEVELS
    }
    out["aurc"] = float((1 - prec).mean())
    return out


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("booking_date", kind="stable")
    cut = df["booking_date"].iloc[int(len(df) * (1 - test_frac))]
    return df[df["booking_date"] < cut], df[df["booking_date"] >= cut]


def run(df: pd.DataFrame, folds: int = 5, models: dict | None = None,
        n_boot: int = 0) -> list[Score]:  # fmt: skip
    models = MODELS if models is None else models
    df = df.dropna(subset=["category"]).reset_index(drop=True)
    df["category"] = collapse_rare(df["category"])
    out: list[Score] = []
    tr, te = temporal_split(df)
    for name, cls in models.items():
        out.append(
            score(name, cls(), tr, tr["category"], te, te["category"], "temporal")
        )
    groups = df["description"].map(merchant_key)
    splits = list(GroupKFold(n_splits=folds).split(df, df["category"], groups))
    for name, cls in models.items():
        parts = [predict_split(cls(), df.iloc[a], df["category"].iloc[a], df.iloc[b],
                               df["category"].iloc[b], "comercios nuevos") for a, b in splits]  # fmt: skip
        # Pliegues concatenados (como cross_val_predict): accuracy y cobertura coinciden con la
        # media ponderada; el F1 macro se calcula sobre todas las predicciones juntas.
        out.append(metrics(name, "comercios nuevos", Preds.concat(parts)))
    if n_boot:
        base = {s.split: s for s in out if s.model == BASELINE}
        for s in out:
            s.ci = bootstrap(s.preds, n_boot)
            if s.split in base and s.model != BASELINE:
                s.vs_baseline = paired_diff(s.preds, base[s.split].preds, n_boot)
    return out


def costs(df: pd.DataFrame, models: dict, single: int = 30,
          max_batch: int | None = None) -> list[dict]:  # fmt: skip
    """Coste de cada modelo en la división temporal, **sin caché de embeddings** (si no, la
    latencia sería falsa): tiempo de ajuste, ms por movimiento en lote, ms de un movimiento
    suelto (mediana) y tamaño (modelo serializado + pesos del codificador congelado).
    `max_batch` limita las filas del lote (en CPU, el LLM tardaría minutos con toda la prueba).
    El dispositivo es el de `embeddings.device()`; se fuerza con `embeddings.use_device`."""
    df = df.dropna(subset=["category"]).reset_index(drop=True)
    df["category"] = collapse_rare(df["category"])
    tr, te = temporal_split(df)
    if max_batch:
        te = te.iloc[:max_batch]
    rows = []
    for name, cls in models.items():
        with embeddings.no_cache():
            m = cls()
            t0 = time.perf_counter()
            m.fit(tr, tr["category"])
            fit_s = time.perf_counter() - t0
            m.predict_proba(te.iloc[:1])  # calentamiento (carga del codificador, CUDA)
            t0 = time.perf_counter()
            m.predict_proba(te)
            batch_ms = (time.perf_counter() - t0) * 1000 / len(te)
            lat = []
            for i in range(min(single, len(te))):
                t0 = time.perf_counter()
                m.predict_proba(te.iloc[[i]])
                lat.append((time.perf_counter() - t0) * 1000)
        size = len(pickle.dumps(m))
        if hasattr(m, "frozen_bytes"):  # pesos congelados del codificador o del LLM
            size += m.frozen_bytes()
        rows.append({"model": name, "fit_s": fit_s, "batch_ms": batch_ms,
                     "single_ms": float(np.median(lat)), "size_mb": size / 1e6})  # fmt: skip
    return rows


def _pct(x) -> str:
    return "—" if x is None or np.isnan(x) else f"{x:.0%}"


def _ci(ci) -> str:
    return "—" if not ci else f"{ci[0]:.0%}–{ci[1]:.0%}"


def report(scores: list[Score], title: str, df: pd.DataFrame,
           cost_tables: list[tuple[str, list[dict]]] = ()) -> str:  # fmt: skip
    """`cost_tables`: [(descripción del dispositivo, filas de `costs`)], una tabla por cada uno."""
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
    for s in scores:
        lines.append(f"| {s.split} | {s.model} | {s.accuracy:.0%} | {s.macro_f1:.2f} | "
                     f"{s.coverage_at_target:.0%} | {_pct(s.precision_covered)} |")  # fmt: skip
    if any(s.ci for s in scores):
        lines += [
            "",
            "## Intervalos de confianza (bootstrap por comercio, 95 %)",
            "",
            (
                "Se remuestrean comercios enteros de la prueba; el umbral no se toca. «Δ vs TF-IDF» "
                "es la diferencia de accuracy frente a TF-IDF + logística en los mismos remuestreos: si "
                "su intervalo incluye el 0, la diferencia no es concluyente."
            ),
            "",
            "| División | Modelo | Accuracy | Cobertura | Precisión cubierta | Δ vs TF-IDF |",
            "|---|---|---|---|---|---|",
        ]
        for s in scores:
            d = s.vs_baseline
            delta = "—" if d is None else f"{d[0]:+.1%} ({d[1]:+.1%} a {d[2]:+.1%})"
            lines.append(f"| {s.split} | {s.model} | {_ci(s.ci['accuracy'])} | "
                         f"{_ci(s.ci['coverage'])} | {_ci(s.ci['precision_covered'])} | {delta} |")  # fmt: skip
    lines += [
        "",
        "## Curva riesgo-cobertura",
        "",
        (
            "Precisión si se categoriza solo el x % de movimientos en que el modelo está más seguro "
            "(descriptivo, sin umbral). AURC = error medio a lo largo de la curva; menor es mejor."
        ),
        "",
        "| División | Modelo | "
        + " | ".join(f"{lvl:.0%}" for lvl in RC_LEVELS)
        + " | AURC |",
        "|---|---|" + "---|" * (len(RC_LEVELS) + 1),
    ]
    for s in scores:
        rc = risk_coverage(s.preds)
        cells = " | ".join(f"{rc[lvl]:.0%}" for lvl in RC_LEVELS)
        lines.append(f"| {s.split} | {s.model} | {cells} | {rc['aurc']:.3f} |")
    for device, cost_rows in cost_tables:
        lines += [
            "",
            f"## Coste (división temporal, sin caché, dispositivo: {device})",
            "",
            "| Modelo | Ajuste (s) | ms/movimiento en lote | ms de un movimiento suelto | Tamaño (MB) |",
            "|---|---|---|---|---|",
        ]
        for r in cost_rows:
            lines.append(f"| {r['model']} | {r['fit_s']:.1f} | {r['batch_ms']:.2f} | "
                         f"{r['single_ms']:.1f} | {r['size_mb']:.1f} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", choices=["synthetic", "personal-os"], default="synthetic"
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-embeddings", action="store_true",
                    help="solo los modelos v1 (sin descargar codificadores)")  # fmt: skip
    ap.add_argument(
        "--fixed-c", action="store_true", help="añade el primer intento v2 (C=10)"
    )
    ap.add_argument(
        "--llm", action="store_true", help="añade el LLM local zero-shot (lento)"
    )
    ap.add_argument(
        "--bootstrap", type=int, default=1000, help="remuestreos; 0 = sin IC"
    )
    ap.add_argument("--no-cpu-cost", action="store_true",
                    help="con GPU, no medir también el coste en CPU")  # fmt: skip
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
    models = dict(MODELS)
    use_emb = not args.no_embeddings and embeddings.available()
    if use_emb:
        models |= embedding_models()
    if use_emb and args.fixed_c:
        models |= fixed_c_embedding_models()
    if use_emb and args.llm:
        models |= llm_models()
    scores = run(df, models=models, n_boot=args.bootstrap)
    cost_models = {k: v for k, v in models.items() if k != "mayoritaria"}
    dev = embeddings.device() if use_emb else "cpu"
    cost_tables = [(dev, costs(df, cost_models))]
    if use_emb and dev != "cpu" and not args.no_cpu_cost:
        with embeddings.use_device("cpu"):
            cpu_rows = costs(df, cost_models, max_batch=CPU_COST_ROWS)
        label = f"cpu, {embeddings.TORCH_THREADS} hilo(s), lote de {CPU_COST_ROWS} movimientos"
        cost_tables.append((label, cpu_rows))
    text = report(scores, title, df.dropna(subset=["category"]), cost_tables)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if args.source == "synthetic":
        print(text)
    else:
        print(f"Informe (solo métricas agregadas) en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
