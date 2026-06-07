# Resumen de Modificaciones: Avance 4 a Avance 5

## 1. Módulos Core (NLP Engine)
Archivos en `src/nlp_core/`:
- `generacion.py` y `pipeline.py`: Se modificaron para incorporar la **Arquitectura de Model Cascading** (enrutamiento dinámico). Ahora el pipeline puede consultar primero a un modelo local (Llama 3.1) y derivar a la nube (GPT-4o) basándose en una autoevaluación de confianza (Faithfulness). Adicionalmente se integró la inyección de seguridad y el **Caché Semántico**.
- `retrieval.py`: Se implementó un sistema de **Caché Global en RAM** para los motores léxicos (BM25, TF-IDF, BoW) y el CrossEncoder. Esto elimina los "Cold Starts" al evitar recalcular las matrices matemáticas en cada consulta, acelerando drásticamente las evaluaciones masivas y reduciendo la latencia de inferencia.
- `evals/evaluador.py`: Se actualizó para implementar métricas RAGAS internamente de forma *custom* (sin depender de la librería externa pesada), calculando las métricas requeridas como Answer Relevancy y Faithfulness directamente con el LLM.
- `telemetria.py`: Se actualizó para soportar el guardado de métricas avanzadas utilizadas por el dashboard.
- `seguridad/guardrails.py` (NUEVO): Se creó para implementar las reglas de seguridad defensiva en los prompts del sistema (prevención de Inyecciones de Prompt).

## 2. Scripts de Laboratorio (Métricas Avanzadas MLOps)
Archivos en `src/lab/`:
- `calcular_delta_ma4.py` (NUEVO): Implementa Remuestreo Bootstrap con 1000 iteraciones para calcular intervalos de confianza (95%) en la mejora de NDCG (Rúbrica MA4).
- `generar_pareto_final.py` (NUEVO): Script analítico que consume los resultados para graficar la Frontera de Pareto (Costo/Latencia vs Precisión).
- `extender_taxonomia_ma6.py` y `exportar_auditoria_manual.py` (NUEVOS): Extienden la evaluación de errores y permiten exportar la taxonomía (Retrieval, Generación, Estructural) a Excel para auditoría manual (Human-in-the-loop).
- `consistencia_eval.py` y `diversidad_eval.py`: Se ajustaron para madurar el cálculo del ECE (Expected Calibration Error) y la Correlación de Pearson en los ensambles híbridos.
- `evaluador_integral.py` y `seguridad_eval.py`: Actualizados para integrar los nuevos módulos de evaluación, incluyendo pruebas de Red Teaming (Jailbreaks).
