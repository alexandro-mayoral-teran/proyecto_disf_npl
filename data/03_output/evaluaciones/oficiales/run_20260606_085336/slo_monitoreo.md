| SLO/Métrica | Objetivo | Fuente | Alerta | Responsable |
| :--- | :--- | :--- | :--- | :--- |
| Disponibilidad | 99.5% piloto | DEP-C | <99.5% semanal | Equipo MLOps/infra |
| Latencia P95 | <3.5s | telemetría percentiles | >3.5s por 2 ventanas | Equipo MLOps |
| Hallucination/Faithfulness | faithfulness >=0.80 o escalar | RAGAS-style judge | caída de 10 pp | Owner funcional + MLOps |
| Costo por consulta | mantener fallback nube <=20% | telemetria_llm.jsonl | fallback >30% | Owner producto |