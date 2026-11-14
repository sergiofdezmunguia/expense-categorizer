# Categorizador de gastos que generaliza a comercios nuevos

Las reglas («MERCADONA → supermercado») resuelven bien lo conocido y nada de lo nuevo. Este
proyecto mide, con una evaluación honesta, **cuánto se puede automatizar con un modelo pequeño
y explicable** y **cuándo debe abstenerse** y dejarlo para revisión humana.

Lo uso con mis movimientos reales de Santander y Trade Republic, que viven en local en
[Personal OS](#privacidad). Este repo solo contiene código, datos **sintéticos** y métricas
agregadas.

## Resultado

Datos reales: 1.542 movimientos etiquetados, 35 categorías, jul-2025 → nov-2026.

| División | Modelo | Accuracy | F1 macro | Cobertura (umbral ≥ 90 %) | Precisión cubierta |
|---|---|---|---|---|---|
| temporal | memoria de comercios (≈ reglas) | 32 % | 0,23 | 32 % | 100 % |
| temporal | **texto (TF-IDF + logística)** | **88 %** | **0,80** | **98 %** | 89 % |
| comercios nuevos | memoria de comercios (≈ reglas) | 1 % | 0,00 | 0 % | — |
| comercios nuevos | **texto (TF-IDF + logística)** | **66 %** | **0,59** | **29 %** | **90 %** |

Cómo leerlo:

- **Temporal** (entrena con el pasado y predice el último 20 %) es el uso diario: el modelo
  categoriza casi todo lo que llega con un 88 % de aciertos.
- **Comercios nuevos** (GroupKFold: ningún comercio de la prueba está en el entrenamiento) es
  donde las reglas no llegan. El modelo acierta 2 de cada 3 y, sobre todo, **sabe cuándo está
  seguro**: el 29 % que se atreve a categorizar solo lo acierta el 90 % de las veces, justo el objetivo fijado.
- La memoria de comercios es casi perfecta cuando responde (100 %), pero responde poco: solo
  recuerda comercios con una categoría consistente (pureza ≥ 90 %). Recordar claves ambiguas,
  como transferencias, bajaba el híbrido del 88 % al 62 %.

## Decisiones de evaluación

- **Dos divisiones**, porque una sola miente: la aleatoria filtra comercios entre entrenamiento
  y prueba y premia memorizar.
- **Abstención con umbral elegido sin mirar la prueba**: validación cruzada interna que imita a
  la prueba (por comercio o aleatoria). La columna «precisión cubierta» es la real.
- **Categorías raras** (< 5 ejemplos) se agrupan en `otros-gastos`: no se pueden ni aprender ni
  evaluar con fiabilidad.
- **Etiquetas**: el 80 % viene de mis reglas, el 20 % de una revisión asistida. Por eso la
  división que importa es la de comercios nuevos: ahí el modelo no puede limitarse a copiar
  las reglas.

## Uso

```bash
uv sync
uv run python -m expense_categorizer.evaluate --source synthetic     # demo reproducible
uv run python -m expense_categorizer.evaluate --source personal-os   # mis datos, en local
uv run pytest
```

Con datos sintéticos (`src/expense_categorizer/synthetic.py`: familias de comercios, comercios
ambiguos como Amazon o PayPal y un 3 % de etiquetas mal puestas), el resultado está en
[`results/synthetic.md`](results/synthetic.md).

## Estructura

| Fichero | Qué hace |
|---|---|
| `text.py` | Normaliza conceptos bancarios y obtiene una clave de comercio aproximada |
| `models.py` | Mayoritaria, memoria de comercios, TF-IDF + logística e híbrido |
| `evaluate.py` | Divisiones, abstención, métricas e informe |
| `synthetic.py` | Generador de movimientos inventados |
| `data.py` | Lectura de solo lectura de la base local de Personal OS |

## Privacidad

Ningún movimiento real sale del PC: `data/`, `*.db`, `*.csv` y `results/real/` están en
`.gitignore`. Los tests y la demo usan datos inventados.

## Siguientes pasos

- Embeddings multilingües pequeños frente a TF-IDF en comercios nuevos.
- Un LLM local en modo zero-shot como referencia, con su coste y su latencia.
- Integrarlo en Personal OS como **propuesta** de categoría (nunca sobre lo puesto a mano),
  con el umbral de abstención como frontera entre automático y revisión.

Nota técnica: en WSL, scikit-learn con todos los hilos de BLAS tardaba 17 s por ajuste frente a
0,4 s con uno (`threadpoolctl`), con resultados idénticos.
