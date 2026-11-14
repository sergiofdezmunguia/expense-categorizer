"""Movimientos bancarios inventados con la forma de los reales (Santander / Trade Republic).

Sirven para la demo pública y los tests: el repo nunca contiene datos reales. Cada categoría
tiene varias «familias» de comercio (p. ej. farmacias con nombres distintos) para que la
evaluación con comercios nuevos tenga sentido: un comercio nuevo se parece a otros de su
categoría, pero no es ninguno de ellos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# categoría -> (prefijos de comercio, nombres propios, importe medio €, signo)
CATALOG = {
    "supermercado": (
        ["MERCADONA", "CARREFOUR EXPRESS", "LIDL", "DIA", "ALCAMPO", "SUPERCOR"],
        35,
        -1,
    ),
    "restaurantes": (["REST", "BAR", "TABERNA", "PIZZERIA", "BURGER", "SUSHI"], 22, -1),
    "comida-a-domicilio": (["GLOVO*", "UBER *EATS", "JUST EAT", "DELIVEROO"], 18, -1),
    "transporte-publico": (
        ["METRO DE MADRID", "RENFE CERCANIAS", "EMT MADRID", "BICIMAD MPASS"],
        8,
        -1,
    ),
    "taxi": (["TAXI", "CABIFY", "UBER *TRIP", "FREENOW"], 14, -1),
    "farmacia": (["FARMACIA", "FCIA", "PARAFARMACIA"], 12, -1),
    "deporte": (["GIMNASIO", "PLAYTOMIC", "DECATHLON", "BASIC FIT"], 30, -1),
    "suscripciones": (
        ["SPOTIFY", "NETFLIX.COM", "APPLE.COM/BILL", "MICROSOFT*", "ANTHROPIC"],
        11,
        -1,
    ),
    "ropa": (["ZARA", "PRIMARK", "PULL&BEAR", "H&M", "MANGO"], 40, -1),
    "ocio": (["CINES", "TEATRO", "TICKETMASTER", "BOWLING", "CONCIERTO"], 25, -1),
    "viajes": (["IBERIA", "RYANAIR", "VUELING", "BOOKING.COM", "AIRBNB"], 140, -1),
    "nomina": (["NOMINA", "TRANSFERENCIA NOMINA"], 2100, 1),
    "intereses": (["INTERESES", "ABONO INTERESES"], 9, 1),
    "inversion": (["PLAN DE INVERSION", "COMPRA ETF", "SAVEBACK"], 300, -1),
}
NAMES = ["LOPEZ", "GARCIA", "DEL SOL", "SAN JUAN", "LA PLAZA", "CENTRO", "NORTE", "OLIVO",
         "MARINA", "ALCALA", "PRADO", "RIVAS", "LUNA", "TRIANA", "MAYOR", "ROSALES"]  # fmt: skip
# Comercios genéricos que aparecen en varias categorías: la ambigüedad real (Amazon, PayPal…).
AMBIGUOUS = {"AMAZON.ES": ["ropa", "ocio", "supermercado"], "PAYPAL *": ["ropa", "ocio", "suscripciones", "viajes"],
             "EL CORTE INGLES": ["ropa", "supermercado", "ocio"], "BIZUM": ["restaurantes", "ocio", "viajes"]}  # fmt: skip
LABEL_NOISE = 0.03  # etiquetas mal puestas, como en la vida real
CITIES = ["MADRID", "BARCELONA", "S/C TENERIFE", "VALENCIA", "SEVILLA", ""]


def generate(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = list(CATALOG)
    weights = np.array([12, 9, 5, 7, 3, 3, 3, 4, 3, 3, 1, 1, 1, 2], dtype=float)
    weights /= weights.sum()
    dates = pd.date_range("2025-07-01", "2026-11-13", freq="D")
    rows = []
    for _ in range(n):
        cat = cats[rng.choice(len(cats), p=weights)]
        prefixes, mean, sign = CATALOG[cat]
        prefix = prefixes[rng.integers(len(prefixes))]
        if rng.random() < 0.12:
            options = [k for k, v in AMBIGUOUS.items() if cat in v]
            if options:
                prefix = options[rng.integers(len(options))]
        name = NAMES[rng.integers(len(NAMES))] if rng.random() < 0.6 else ""
        noise = f"{rng.integers(1000, 9999)}" if rng.random() < 0.5 else ""
        city = CITIES[rng.integers(len(CITIES))]
        desc = " ".join(p for p in ["COMPRA TARJ. 5402XXXXXXXX" if rng.random() < 0.3 else "",
                                    prefix, name, noise, city] if p)  # fmt: skip
        amount = sign * max(0.5, rng.lognormal(np.log(mean), 0.5))
        rows.append(
            {
                "booking_date": dates[rng.integers(len(dates))].date().isoformat(),
                "description": desc,
                "amount_cents": round(amount * 100),
                "account_kind": "broker"
                if cat in ("intereses", "inversion")
                else "bank",
                "category": cats[rng.integers(len(cats))]
                if rng.random() < LABEL_NOISE
                else cat,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("booking_date", kind="stable")
        .reset_index(drop=True)
    )
