# Categorizador de gastos que generaliza a comercios nuevos

Las reglas («MERCADONA → supermercado») resuelven bien lo conocido y nada de lo nuevo. Este
proyecto mide, con una evaluación honesta, **cuánto se puede automatizar con un modelo pequeño
y explicable** y **cuándo debe abstenerse** y dejarlo para revisión humana.

Lo uso con mis movimientos reales de Santander y Trade Republic, que viven en local en
[Personal OS](#privacidad). Este repo solo contiene código, datos **sintéticos** y métricas
agregadas.

## Resultado (v1)

Datos reales: 1.542 movimientos etiquetados, 35 categorías, jul-2025 → nov-2026.

| División | Modelo | Accuracy | F1 macro | Cobertura (umbral ≥ 90 %) | Precisión cubierta |
|---|---|---|---|---|---|
| temporal | memoria de comercios (≈ reglas) | 32 % | 0,23 | 32 % | 100 % |
| temporal | **texto (TF-IDF + logística)** | **88 %** | **0,80** | **98 %** | 89 % |
| comercios nuevos | memoria de comercios (≈ reglas) | 1 % | 0,00 | 0 % | — |
| comercios nuevos | **texto (TF-IDF + logística)** | **66 %** | **0,47** | **29 %** | **90 %** |

El F1 macro de comercios nuevos se calcula sobre las predicciones de los cinco pliegues juntas
(la versión anterior promediaba el F1 de cada pliegue y daba 0,59; no son comparables).

Cómo leerlo:

- **Temporal** (entrena con el pasado y predice el último 20 %) es el uso diario: el modelo
  categoriza casi todo lo que llega con un 88 % de aciertos.
- **Comercios nuevos** (GroupKFold: ningún comercio de la prueba está en el entrenamiento) es
  donde las reglas no llegan. El modelo acierta 2 de cada 3 y, sobre todo, **sabe cuándo está
  seguro**: el 29 % que se atreve a categorizar solo lo acierta el 90 % de las veces, justo el objetivo fijado.
- La memoria de comercios es casi perfecta cuando responde (100 %), pero responde poco: solo
  recuerda comercios con una categoría consistente (pureza ≥ 90 %). Recordar claves ambiguas,
  como transferencias, bajaba el híbrido del 88 % al 62 %.

## v2: ¿ayudan los embeddings o un LLM local?

Misma evaluación, añadiendo dos codificadores multilingües pequeños (multilingual-e5-small y
paraphrase-multilingual-MiniLM-L12, ~118 M parámetros, congelados), TF-IDF + e5 concatenados,
un zero-shot por similitud con el nombre de la categoría y un LLM local en zero-shot
(Qwen2.5-1.5B-Instruct, opcional). Modelos y combinaciones declarados antes de ver resultados.

**Comercios nuevos, datos reales** (Δ = diferencia de accuracy frente a TF-IDF v1, bootstrap
pareado por comercio, IC 95 %):

| Modelo | Accuracy | Cobertura | Precisión cubierta | Δ vs TF-IDF v1 |
|---|---|---|---|---|
| TF-IDF + logística (v1, C = 10) | 66 % | 29 % | 90 % | — |
| TF-IDF + logística, C interno | **70 %** | **41 %** | **98 %** | +3,6 (−0,7 a +5,9) |
| TF-IDF + e5-small, C interno | 70 % | 25 % | 100 % | +3,9 (+1,2 a +6,0) |
| e5-small + logística, C interno | 42 % | 1 % | 91 % | −24,2 (−44,1 a −11,9) |
| MiniLM-L12 + logística, C interno | 37 % | 4 % | 42 % | −29,1 (−56,4 a −15,6) |
| *primer intento:* e5-small, C = 10 | 32 % | 0 % | — | −33,7 (−48,1 a −23,4) |
| LLM zero-shot (Qwen2.5-1.5B) | 30 % | 2 % | 78 % | −35,8 (−64,7 a −21,0) |
| zero-shot por similitud (e5) | 9 % | 2 % | 48 % | −57,1 (−80,6 a −40,0) |

Conclusiones:

- **Los embeddings no aportan sobre TF-IDF con mis conceptos bancarios.** Solos pierden más de
  20 puntos; concatenados a TF-IDF empatan con TF-IDF bien regularizado (70 % frente a 70 %) y
  cubren menos. La mejora significativa de TF-IDF + e5 frente a v1 viene de elegir bien C, no
  del embedding: TF-IDF solo con C interno consigue lo mismo.
- **Corrección honesta del primer intento.** Con C = 10 fijo (el de TF-IDF), los embeddings
  infraajustaban (90 % de aciertos en su propio entrenamiento) y daban un 32 %. Elegir C con
  validación interna agrupada por comercio, también para TF-IDF como control, sube e5 al 42 %:
  mejor, pero lejos. El informe conserva ambas cifras (`--fixed-c`).
- **Lo que sí sale de v2:** TF-IDF con C interno es la nueva referencia. Categoriza solo el
  41 % de los movimientos de comercios nuevos con un 98 % de precisión (antes, 29 % con un 90 %).
- **Zero-shot, ni con LLM.** Sin ejemplos, nombres de categoría como `otros-gastos` o mis
  categorías propias no se adivinan: 30 % con un LLM de 1,5 B, a 80 ms por movimiento en GPU y
  ~11 s en CPU.
- En datos sintéticos todos los modelos entrenados empatan (~88 %): el generador comparte
  trozos de palabra entre comercios de la misma familia, justo lo que TF-IDF aprovecha. Por
  eso la conclusión se apoya en los datos reales.
- Los intervalos son anchos (182 comercios): solo son concluyentes las diferencias grandes.

**Coste** (datos reales, división temporal, sin caché):

| Modelo | ms por movimiento suelto, GPU | ídem, CPU (1 hilo) | Tamaño |
|---|---|---|---|
| TF-IDF + logística | 7 | 7 | 3 MB |
| e5-small + logística | 14 | 26 | 471 MB |
| TF-IDF + e5-small | 17 | 30 | 474 MB |
| LLM zero-shot (Qwen2.5-1.5B) | 80 | ~11.400 | 3,1 GB (fp16) / 6,2 GB (fp32) |

Informe sintético completo, con curvas riesgo-cobertura, en
[`results/synthetic.md`](results/synthetic.md).

## Decisiones de evaluación

- **Dos divisiones**, porque una sola miente: la aleatoria filtra comercios entre entrenamiento
  y prueba y premia memorizar.
- **Abstención con umbral elegido sin mirar la prueba**: validación cruzada interna que imita a
  la prueba (por comercio o aleatoria). La columna «precisión cubierta» es la real.
- **Hiperparámetros sin mirar la prueba**: en v2, la C de la logística se elige con
  validación cruzada agrupada por comercio dentro del entrenamiento (`TunedC`).
- **Intervalos honestos**: bootstrap por comercio (los movimientos de un mismo comercio no son
  independientes) y Δ pareado frente a la referencia en los mismos remuestreos.
- **Categorías raras** (< 5 ejemplos) se agrupan en `otros-gastos`: no se pueden ni aprender ni
  evaluar con fiabilidad.
- **Etiquetas**: el 80 % viene de mis reglas, el 20 % de una revisión asistida. Por eso la
  división que importa es la de comercios nuevos: ahí el modelo no puede limitarse a copiar
  las reglas.

## Uso

```bash
uv sync
uv run python -m expense_categorizer.evaluate --source synthetic     # v1 (sin el extra)
uv run python -m expense_categorizer.evaluate --source personal-os   # mis datos, en local
uv run pytest

# v2: embeddings (descarga ~0,5 GB por codificador a ~/.cache/huggingface)
uv sync --extra embeddings
uv run --extra embeddings python -m expense_categorizer.evaluate --fixed-c   # + primer intento
uv run --extra embeddings python -m expense_categorizer.evaluate --llm       # + Qwen2.5-1.5B (~3 GB)
EC_TEST_LLM=1 uv run --extra embeddings pytest                                  # incluye el test del LLM
```

Todo corre en local. torch se fija a las ruedas de CUDA 12.6 (índice de PyTorch, solo en el
venv) porque las de PyPI piden un driver más nuevo que el de este WSL; sin GPU, usa la CPU. Los
embeddings se cachean solo en memoria: con datos reales, no quedan vectores derivados en disco.

Con datos sintéticos (`src/expense_categorizer/synthetic.py`: familias de comercios, comercios
ambiguos como Amazon o PayPal y un 3 % de etiquetas mal puestas), el resultado está en
[`results/synthetic.md`](results/synthetic.md).

## Estructura

| Fichero | Qué hace |
|---|---|
| `text.py` | Normaliza conceptos bancarios y obtiene una clave de comercio aproximada |
| `models.py` | Mayoritaria, memoria de comercios, TF-IDF + logística, híbrido, embeddings, zero-shot y `TunedC` |
| `embeddings.py` | Codificadores de frases locales, caché en memoria, CPU/GPU |
| `llm.py` | LLM local zero-shot que puntúa el nombre de cada categoría |
| `evaluate.py` | Divisiones, abstención, bootstrap, riesgo-cobertura, coste e informe |
| `synthetic.py` | Generador de movimientos inventados |
| `data.py` | Lectura de solo lectura de la base local de Personal OS |

## Privacidad

Ningún movimiento real sale del PC: `data/`, `*.db`, `*.csv` y `results/real/` están en
`.gitignore`. Los tests y la demo usan datos inventados.

## Siguientes pasos

- Integrarlo en Personal OS como **propuesta** de categoría (nunca sobre lo puesto a mano),
  con TF-IDF + C interno y el umbral de abstención como frontera entre automático y revisión.
- Si se reintenta con embeddings: afinarlos (no congelados) o pocos ejemplos en el prompt del
  LLM; con 182 comercios, cualquier ganancia pequeña quedará dentro del ruido.

Nota técnica: en WSL, scikit-learn con todos los hilos de BLAS tardaba 17 s por ajuste frente a
0,4 s con uno (`threadpoolctl`), con resultados idénticos.
