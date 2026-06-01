| Candidato | NDCG nube | Latencia promedio nube (s) | P95 proxy nube (s) | Multiplicador latencia vs ganador | Costo/1k nube (USD) | Lectura operativa |
| --- | --- | --- | --- | --- | --- | --- |
| 1_Baseline_Léxico | 0.7140 | 0.303 | 0.409 | 1.082 | 0.510 | Alternativa viable, pero dominada por el baseline semantico. |
| 2_Baseline_Semántico | 0.8457 | 0.280 | 0.378 | 1.000 | 0.507 | Mejor punto operativo: mayor NDCG nube y menor latencia. |
| 3_Híbrido_Simple | 0.8261 | 0.508 | 0.685 | 1.812 | 0.511 | Alternativa viable, pero dominada por el baseline semantico. |
| 4_Híbrido_Reranker | 0.8301 | 2.824 | 3.812 | 10.079 | 0.506 | Complejidad costosa en latencia sin lift estadistico claro. |
| 5_Semántico_Expandido | 0.8439 | 4.837 | 6.530 | 17.264 | 0.511 | Complejidad costosa en latencia sin lift estadistico claro. |
| 6_SOTA_Completo | 0.8407 | 9.377 | 12.659 | 33.466 | 0.502 | Ruta de escalamiento: calidad cercana, pero latencia muy superior. |
