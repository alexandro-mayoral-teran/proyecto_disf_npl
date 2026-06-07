| Métrica estilo RAGAS | Cómo se aproxima | Valor observado | Uso operativo |
| :--- | :--- | :--- | :--- |
| Faithfulness | generacion_exitosa + evidencia en contexto | 0.2385 | Ruteo cascade y alerta de alucinación |
| Context precision/retrieval | retrieval_exitoso | 0.5688 | Detectar fallos de chunking, expansión o top-k |
| Formato/answer validity | formato_exitoso | 1.0 | Monitoreo de respuestas aptas para consumo por analista/app |