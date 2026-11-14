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
