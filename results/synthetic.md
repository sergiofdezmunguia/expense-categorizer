# Resultados con datos sintéticos

2000 movimientos etiquetados, 14 categorías, 612 comercios (clave aproximada).

Cobertura = parte que se categoriza sola con un umbral elegido para ≥ 90% de precisión en validación interna; «precisión cubierta» es la que se obtiene de verdad en la prueba.

| División | Modelo | Accuracy | F1 macro | Cobertura | Precisión cubierta |
|---|---|---|---|---|---|
| temporal | mayoritaria | 20% | 0.02 | 0% | — |
| temporal | memoria de comercios (≈ reglas) | 65% | 0.60 | 61% | 96% |
| temporal | texto (TF-IDF + logística) | 93% | 0.87 | 100% | 93% |
| temporal | híbrido (memoria + texto) | 93% | 0.87 | 100% | 93% |
| temporal | zero-shot (similitud e5 con el nombre) | 43% | 0.37 | 12% | 85% |
| temporal | TF-IDF + logística, C interno | 95% | 0.91 | 100% | 95% |
| temporal | e5-small + logística, C interno | 94% | 0.89 | 100% | 94% |
| temporal | MiniLM-L12 + logística, C interno | 91% | 0.86 | 100% | 91% |
| temporal | TF-IDF + e5-small + logística, C interno | 94% | 0.90 | 100% | 95% |
| temporal | e5-small + logística, C=10 | 95% | 0.90 | 100% | 95% |
| temporal | MiniLM-L12 + logística, C=10 | 91% | 0.87 | 99% | 92% |
| temporal | TF-IDF + e5-small + logística, C=10 | 94% | 0.88 | 100% | 94% |
| temporal | LLM zero-shot (Qwen2.5-1.5B) | 51% | 0.34 | 14% | 84% |
| comercios nuevos | mayoritaria | 21% | 0.02 | 0% | — |
| comercios nuevos | memoria de comercios (≈ reglas) | 21% | 0.02 | 0% | — |
| comercios nuevos | texto (TF-IDF + logística) | 88% | 0.87 | 95% | 92% |
| comercios nuevos | híbrido (memoria + texto) | 88% | 0.87 | 95% | 92% |
| comercios nuevos | zero-shot (similitud e5 con el nombre) | 45% | 0.42 | 11% | 88% |
| comercios nuevos | TF-IDF + logística, C interno | 88% | 0.87 | 95% | 92% |
| comercios nuevos | e5-small + logística, C interno | 88% | 0.87 | 86% | 92% |
| comercios nuevos | MiniLM-L12 + logística, C interno | 86% | 0.84 | 69% | 94% |
| comercios nuevos | TF-IDF + e5-small + logística, C interno | 88% | 0.87 | 94% | 93% |
| comercios nuevos | e5-small + logística, C=10 | 89% | 0.88 | 94% | 91% |
| comercios nuevos | MiniLM-L12 + logística, C=10 | 84% | 0.83 | 80% | 93% |
| comercios nuevos | TF-IDF + e5-small + logística, C=10 | 88% | 0.87 | 95% | 92% |
| comercios nuevos | LLM zero-shot (Qwen2.5-1.5B) | 48% | 0.34 | 15% | 89% |

## Intervalos de confianza (bootstrap por comercio, 95 %)

Se remuestrean comercios enteros de la prueba; el umbral no se toca. «Δ vs TF-IDF» es la diferencia de accuracy frente a TF-IDF + logística en los mismos remuestreos: si su intervalo incluye el 0, la diferencia no es concluyente.

| División | Modelo | Accuracy | Cobertura | Precisión cubierta | Δ vs TF-IDF |
|---|---|---|---|---|---|
| temporal | mayoritaria | 13%–28% | 0%–0% | — | -73.6% (-81.0% a -64.8%) |
| temporal | memoria de comercios (≈ reglas) | 58%–72% | 53%–69% | 94%–99% | -28.1% (-35.3% a -20.9%) |
| temporal | texto (TF-IDF + logística) | 90%–96% | 100%–100% | 90%–96% | — |
| temporal | híbrido (memoria + texto) | 90%–96% | 100%–100% | 90%–96% | +0.0% (+0.0% a +0.0%) |
| temporal | zero-shot (similitud e5 con el nombre) | 36%–51% | 7%–18% | 75%–95% | -49.9% (-56.7% a -43.3%) |
| temporal | TF-IDF + logística, C interno | 90%–97% | 99%–100% | 90%–97% | +1.2% (-0.5% a +2.9%) |
| temporal | e5-small + logística, C interno | 90%–97% | 100%–100% | 90%–97% | +0.5% (-0.8% a +1.9%) |
| temporal | MiniLM-L12 + logística, C interno | 87%–94% | 100%–100% | 87%–94% | -2.5% (-4.5% a -0.8%) |
| temporal | TF-IDF + e5-small + logística, C interno | 90%–97% | 99%–100% | 90%–97% | +1.0% (-0.7% a +2.5%) |
| temporal | e5-small + logística, C=10 | 91%–97% | 100%–100% | 91%–97% | +1.2% (-0.5% a +3.1%) |
| temporal | MiniLM-L12 + logística, C=10 | 88%–94% | 98%–100% | 88%–95% | -2.0% (-4.1% a +0.0%) |
| temporal | TF-IDF + e5-small + logística, C=10 | 91%–97% | 100%–100% | 91%–97% | +0.5% (+0.0% a +1.4%) |
| temporal | LLM zero-shot (Qwen2.5-1.5B) | 43%–60% | 8%–21% | 73%–93% | -42.0% (-50.5% a -32.9%) |
| comercios nuevos | mayoritaria | 15%–27% | 0%–0% | — | -67.2% (-73.7% a -60.1%) |
| comercios nuevos | memoria de comercios (≈ reglas) | 15%–27% | 0%–0% | — | -67.2% (-73.7% a -60.1%) |
| comercios nuevos | texto (TF-IDF + logística) | 82%–92% | 91%–99% | 89%–94% | — |
| comercios nuevos | híbrido (memoria + texto) | 82%–92% | 91%–99% | 89%–94% | +0.0% (+0.0% a +0.0%) |
| comercios nuevos | zero-shot (similitud e5 con el nombre) | 40%–50% | 8%–14% | 83%–93% | -43.0% (-50.0% a -34.6%) |
| comercios nuevos | TF-IDF + logística, C interno | 82%–92% | 91%–99% | 89%–94% | +0.0% (-0.4% a +0.4%) |
| comercios nuevos | e5-small + logística, C interno | 83%–92% | 81%–91% | 89%–95% | +0.2% (-1.4% a +2.1%) |
| comercios nuevos | MiniLM-L12 + logística, C interno | 82%–89% | 63%–74% | 91%–95% | -1.9% (-5.5% a +2.2%) |
| comercios nuevos | TF-IDF + e5-small + logística, C interno | 82%–92% | 89%–98% | 90%–94% | -0.1% (-0.6% a +0.2%) |
| comercios nuevos | e5-small + logística, C=10 | 84%–92% | 90%–97% | 88%–94% | +0.9% (-0.8% a +2.9%) |
| comercios nuevos | MiniLM-L12 + logística, C=10 | 79%–88% | 75%–83% | 90%–95% | -3.6% (-7.5% a +0.7%) |
| comercios nuevos | TF-IDF + e5-small + logística, C=10 | 82%–92% | 91%–99% | 89%–94% | -0.1% (-0.4% a +0.2%) |
| comercios nuevos | LLM zero-shot (Qwen2.5-1.5B) | 41%–54% | 9%–20% | 83%–94% | -40.5% (-49.7% a -30.6%) |

## Curva riesgo-cobertura

Precisión si se categoriza solo el x % de movimientos en que el modelo está más seguro (descriptivo, sin umbral). AURC = error medio a lo largo de la curva; menor es mejor.

| División | Modelo | 25% | 50% | 75% | AURC |
|---|---|---|---|---|---|
| temporal | mayoritaria | 25% | 21% | 19% | 0.777 |
| temporal | memoria de comercios (≈ reglas) | 98% | 96% | 81% | 0.103 |
| temporal | texto (TF-IDF + logística) | 97% | 98% | 97% | 0.034 |
| temporal | híbrido (memoria + texto) | 98% | 96% | 97% | 0.031 |
| temporal | zero-shot (similitud e5 con el nombre) | 80% | 62% | 51% | 0.349 |
| temporal | TF-IDF + logística, C interno | 96% | 97% | 96% | 0.043 |
| temporal | e5-small + logística, C interno | 98% | 98% | 97% | 0.036 |
| temporal | MiniLM-L12 + logística, C interno | 97% | 97% | 97% | 0.041 |
| temporal | TF-IDF + e5-small + logística, C interno | 96% | 97% | 96% | 0.041 |
| temporal | e5-small + logística, C=10 | 97% | 97% | 97% | 0.041 |
| temporal | MiniLM-L12 + logística, C=10 | 95% | 97% | 96% | 0.048 |
| temporal | TF-IDF + e5-small + logística, C=10 | 97% | 98% | 97% | 0.034 |
| temporal | LLM zero-shot (Qwen2.5-1.5B) | 85% | 67% | 59% | 0.296 |
| comercios nuevos | mayoritaria | 14% | 16% | 17% | 0.844 |
| comercios nuevos | memoria de comercios (≈ reglas) | 14% | 16% | 17% | 0.844 |
| comercios nuevos | texto (TF-IDF + logística) | 96% | 97% | 97% | 0.039 |
| comercios nuevos | híbrido (memoria + texto) | 96% | 97% | 97% | 0.039 |
| comercios nuevos | zero-shot (similitud e5 con el nombre) | 81% | 66% | 54% | 0.325 |
| comercios nuevos | TF-IDF + logística, C interno | 97% | 97% | 96% | 0.040 |
| comercios nuevos | e5-small + logística, C interno | 97% | 96% | 95% | 0.046 |
| comercios nuevos | MiniLM-L12 + logística, C interno | 96% | 95% | 93% | 0.061 |
| comercios nuevos | TF-IDF + e5-small + logística, C interno | 97% | 97% | 96% | 0.040 |
| comercios nuevos | e5-small + logística, C=10 | 97% | 97% | 96% | 0.046 |
| comercios nuevos | MiniLM-L12 + logística, C=10 | 96% | 96% | 94% | 0.062 |
| comercios nuevos | TF-IDF + e5-small + logística, C=10 | 96% | 97% | 97% | 0.040 |
| comercios nuevos | LLM zero-shot (Qwen2.5-1.5B) | 84% | 66% | 57% | 0.298 |

## Coste (división temporal, sin caché, dispositivo: cuda)

| Modelo | Ajuste (s) | ms/movimiento en lote | ms de un movimiento suelto | Tamaño (MB) |
|---|---|---|---|---|
| memoria de comercios (≈ reglas) | 0.2 | 0.02 | 1.4 | 0.0 |
| texto (TF-IDF + logística) | 0.5 | 0.05 | 6.0 | 0.3 |
| híbrido (memoria + texto) | 0.6 | 0.08 | 8.8 | 0.3 |
| zero-shot (similitud e5 con el nombre) | 0.0 | 0.79 | 21.1 | 470.6 |
| TF-IDF + logística, C interno | 4.9 | 0.05 | 6.5 | 0.3 |
| e5-small + logística, C interno | 9.2 | 0.18 | 13.3 | 470.7 |
| MiniLM-L12 + logística, C interno | 8.4 | 0.18 | 13.3 | 470.7 |
| TF-IDF + e5-small + logística, C interno | 14.8 | 0.23 | 19.5 | 471.0 |
| e5-small + logística, C=10 | 0.9 | 0.19 | 15.0 | 470.7 |
| MiniLM-L12 + logística, C=10 | 0.7 | 0.20 | 15.3 | 470.7 |
| TF-IDF + e5-small + logística, C=10 | 1.4 | 0.22 | 18.6 | 471.0 |
| LLM zero-shot (Qwen2.5-1.5B) | 0.0 | 63.93 | 65.9 | 3087.4 |

## Coste (división temporal, sin caché, dispositivo: cpu, 1 hilo(s), lote de 100 movimientos (10 para el LLM))

| Modelo | Ajuste (s) | ms/movimiento en lote | ms de un movimiento suelto | Tamaño (MB) |
|---|---|---|---|---|
| memoria de comercios (≈ reglas) | 0.2 | 0.03 | 1.3 | 0.0 |
| texto (TF-IDF + logística) | 0.5 | 0.11 | 6.3 | 0.3 |
| híbrido (memoria + texto) | 0.6 | 0.16 | 8.6 | 0.3 |
| zero-shot (similitud e5 con el nombre) | 0.0 | 6.42 | 72.9 | 470.6 |
| TF-IDF + logística, C interno | 4.8 | 0.11 | 6.4 | 0.3 |
| e5-small + logística, C interno | 81.9 | 5.79 | 23.5 | 470.7 |
| MiniLM-L12 + logística, C interno | 65.2 | 4.75 | 22.6 | 470.7 |
| TF-IDF + e5-small + logística, C interno | 87.4 | 5.84 | 26.2 | 471.0 |
| e5-small + logística, C=10 | 6.3 | 5.77 | 23.7 | 470.7 |
| MiniLM-L12 + logística, C=10 | 4.9 | 4.85 | 22.0 | 470.7 |
| TF-IDF + e5-small + logística, C=10 | 7.0 | 5.97 | 26.5 | 471.0 |
| LLM zero-shot (Qwen2.5-1.5B) | 0.0 | 6415.48 | 6482.9 | 6174.9 |
