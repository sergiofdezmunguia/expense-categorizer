import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from expense_categorizer import evaluate as ev
from expense_categorizer.models import Hybrid, MerchantMemory, TextModel
from expense_categorizer.synthetic import generate
from expense_categorizer.text import clean, fold, merchant_key


def test_text_normalization_removes_noise_and_accents():
    assert fold("Café Ñoño, S.L.") == "cafe nono s l"
    cleaned = clean("COMPRA TARJ. 5402XXXXXXXX FARMACIA LÓPEZ 1234 MADRID")
    assert "farmacia lopez" in cleaned and not any(ch.isdigit() for ch in cleaned)
    assert "madrid" not in cleaned and "compra" not in cleaned
    assert merchant_key("MERCADONA 1234 MADRID") == merchant_key("Mercadona  madrid")


def test_pick_threshold_reaches_target_or_covers_nothing():
    conf = np.array([0.99, 0.95, 0.9, 0.6, 0.5])
    correct = np.array([True, True, True, False, False])
    assert ev.pick_threshold(conf, correct, 0.9) == 0.9
    assert ev.pick_threshold(conf, np.zeros(5, dtype=bool), 0.9) > 1


def test_memory_only_keeps_pure_merchants():
    df = pd.DataFrame(
        {
            "booking_date": ["2026-01-01"] * 6,
            "description": ["MERCADONA"] * 3 + ["TRANSFERENCIA"] * 3,
            "amount_cents": [-1000] * 6,
            "category": ["supermercado"] * 3 + ["ocio", "viajes", "ropa"],
        }
    )
    m = MerchantMemory().fit(df, df["category"])
    assert list(m.memory_) == ["mercadona"]


def test_grouped_split_never_shares_merchants():
    df = generate(n=300)
    groups = df["description"].map(merchant_key)
    for tr, te in GroupKFold(n_splits=5).split(df, df["category"], groups):
        assert not set(groups.iloc[tr]) & set(groups.iloc[te])


def test_text_model_generalizes_to_unseen_merchant():
    df = generate(n=600)
    model = TextModel().fit(df, df["category"])
    new = pd.DataFrame(
        {
            "booking_date": ["2026-11-01"],
            "description": ["FARMACIA ZURBARAN 7788"],
            "amount_cents": [-1250],
            "account_kind": ["bank"],
        }
    )
    assert model.predict(new)[0] == "farmacia"
    assert Hybrid().fit(df, df["category"]).predict(new)[0] == "farmacia"


def test_run_on_synthetic_beats_baselines():
    scores = {(s.split, s.model): s for s in ev.run(generate(n=500), folds=3)}
    text = scores[("comercios nuevos", "texto (TF-IDF + logística)")]
    memory = scores[("comercios nuevos", "memoria de comercios (≈ reglas)")]
    assert text.accuracy > memory.accuracy + 0.3


def _preds(hits, conf, groups, covered=None):
    hits = np.asarray(hits, dtype=bool)
    y = np.array(["a"] * len(hits))
    return ev.Preds(y_true=y, pred=np.where(hits, "a", "b"), conf=np.asarray(conf, float),
                    covered=np.ones(len(hits), bool) if covered is None else np.asarray(covered),
                    groups=np.asarray(groups))  # fmt: skip


def test_risk_coverage_orders_by_confidence():
    p = _preds([True, True, False, False], [0.9, 0.8, 0.7, 0.6], list("abcd"))
    rc = ev.risk_coverage(p)
    assert rc[0.25] == 1.0 and rc[0.5] == 1.0 and rc[0.75] == 2 / 3
    assert abs(rc["aurc"] - np.mean([0, 0, 1 / 3, 1 / 2])) < 1e-12


def test_bootstrap_resamples_whole_merchants():
    # Dos comercios: uno siempre acierta y otro siempre falla. Por comercio, solo hay
    # remuestreos con 0, 1/2 o 1 de accuracy (con tamaños 3 y 1: 0, 1/4, 3/4 o 1).
    p = _preds([True, True, True, False], [0.9] * 4, ["x", "x", "x", "y"])
    ci = ev.bootstrap(p, n=200)
    assert ci["accuracy"][0] >= 0 and ci["accuracy"][1] <= 1
    accs = {round(p.pred[i].tolist().count("a") / len(i), 4)
            for i in ev._cluster_resamples(p.groups, 200)}  # fmt: skip
    assert accs <= {0.0, 0.25, 0.75, 1.0}


def test_paired_diff_is_zero_against_itself():
    p = _preds([True, False, True], [0.9, 0.5, 0.7], ["x", "y", "z"])
    assert ev.paired_diff(p, p, n=50) == (0.0, 0.0, 0.0)


def test_pooled_folds_keep_test_rows_and_threshold_from_train_only(monkeypatch):
    seen = []
    real = ev.pick_threshold

    def spy(conf, correct, target=ev.TARGET_PRECISION):
        seen.append(len(conf))
        return real(conf, correct, target)

    monkeypatch.setattr(ev, "pick_threshold", spy)
    df = generate(n=300)
    scores = {(s.split, s.model): s for s in ev.run(df, folds=3)}
    s = scores[("comercios nuevos", "texto (TF-IDF + logística)")]
    assert s.n_test == len(df)
    # Cada umbral se elige con filas de entrenamiento (nunca con las 300 de la prueba agregada).
    assert seen and max(seen) < len(df)


emb = __import__("pytest").importorskip("expense_categorizer.embeddings")


@__import__("pytest").mark.skipif(not emb.available(), reason="sin extra embeddings")
def test_embedding_models_generalize_and_cache():
    from expense_categorizer.models import EmbeddingModel

    df = generate(n=600)
    new = pd.DataFrame({"booking_date": ["2026-11-01"], "description": ["FARMACIA ZURBARAN 7788"],
                        "amount_cents": [-1250], "account_kind": ["bank"]})  # fmt: skip
    for text in ("emb", "tfidf+emb"):
        m = EmbeddingModel(model_name=emb.E5_SMALL, text=text).fit(df, df["category"])
        assert m.predict(new)[0] == "farmacia"
    a = emb.encode(emb.E5_SMALL, ["farmacia lopez", "farmacia lopez"])
    with emb.no_cache():
        b = emb.encode(emb.E5_SMALL, ["farmacia lopez"])
    assert np.allclose(a[0], a[1]) and np.allclose(a[0], b[0], atol=1e-5)
    assert abs(np.linalg.norm(a[0]) - 1) < 1e-4


@__import__("pytest").mark.skipif(
    __import__("os").environ.get("EC_TEST_LLM") != "1",
    reason="descarga ~3 GB; EC_TEST_LLM=1",
)
def test_llm_zero_shot_returns_distribution_over_training_labels():
    from expense_categorizer.llm import ZeroShotLLM

    df = generate(n=200)
    m = ZeroShotLLM().fit(df, df["category"])
    p = m.predict_proba(df.iloc[:5])
    assert p.shape == (5, df["category"].nunique()) and np.allclose(p.sum(axis=1), 1)


def test_tuned_c_picks_from_grid_using_only_training_rows():
    from expense_categorizer.models import TunedC

    df = generate(n=300)
    m = TunedC(TextModel(), grid=(0.01, 10.0)).fit(df, df["category"])
    assert m.C_ == 10.0  # C muy pequeña infraajusta en validación interna
    assert m.predict(df.iloc[:3]).shape == (3,)


@__import__("pytest").mark.skipif(not emb.available(), reason="sin extra embeddings")
def test_use_device_forces_cpu_and_restores():
    before = emb.device()
    with emb.use_device("cpu"):
        assert emb.device() == "cpu"
        v = emb.encode(emb.E5_SMALL, ["farmacia lopez"])
    assert emb.device() == before and v.shape == (1, 384)


def test_report_writes_one_cost_table_per_device():
    df = generate(n=300)
    scores = ev.run(df, folds=3, models={"texto (TF-IDF + logística)": TextModel})
    row = {
        "model": "m",
        "fit_s": 1.0,
        "batch_ms": 0.1,
        "single_ms": 2.0,
        "size_mb": 0.3,
    }
    text = ev.report(scores, "t", df, [("cuda", [row]), ("cpu, 1 hilo(s)", [row])])
    assert "dispositivo: cuda)" in text and "dispositivo: cpu, 1 hilo(s))" in text
