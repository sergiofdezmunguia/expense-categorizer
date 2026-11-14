# Resultados con datos sintéticos

2000 movimientos etiquetados, 14 categorías, 612 comercios (clave aproximada).

Cobertura = parte que se categoriza sola con un umbral elegido para ≥ 90% de precisión en validación interna; «precisión cubierta» es la que se obtiene de verdad en la prueba.

| División | Modelo | Accuracy | F1 macro | Cobertura | Precisión cubierta |
|---|---|---|---|---|---|
| temporal | mayoritaria | 20% | 0.02 | 0% | — |
| temporal | memoria de comercios (≈ reglas) | 65% | 0.60 | 61% | 96% |
| temporal | texto (TF-IDF + logística) | 93% | 0.87 | 100% | 93% |
| temporal | híbrido (memoria + texto) | 93% | 0.87 | 100% | 93% |
| comercios nuevos | mayoritaria | 21% | 0.02 | 0% | — |
| comercios nuevos | memoria de comercios (≈ reglas) | 21% | 0.02 | 0% | — |
| comercios nuevos | texto (TF-IDF + logística) | 88% | 0.87 | 95% | 92% |
| comercios nuevos | híbrido (memoria + texto) | 88% | 0.87 | 95% | 92% |
