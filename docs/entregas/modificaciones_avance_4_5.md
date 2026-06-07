# Resumen de Modificaciones: Avance 4 a Avance 5

El presente documento resume los archivos clave que sufrieron modificaciones o fueron creados durante la transición del **Avance 4** al **Avance 5**, así como el motivo técnico u operativo detrás de dichos cambios.

## 1. Módulos Core (NLP Engine)
Archivos en `src/nlp_core/`:
- `generacion.py` y `pipeline.py`: Se modificaron para incorporar la **Arquitectura de Model Cascading** (enrutamiento dinámico). Ahora el pipeline puede consultar primero a un modelo local (Llama 3.1) y derivar a la nube (GPT-4o) basándose en una autoevaluación de confianza (Faithfulness). Adicionalmente se integró la inyección de seguridad y el **Caché Semántico**.
- `retrieval.py`: Se pulió para garantizar la captura de telemetría y soportar las expansiones necesarias para el RAG.
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

## 3. Interfaces de Usuario y API
- `api/main_api.py` y `app/main.js`: Se implementaron llamadas directas y actualizaciones para reflejar el estado real del Model Cascading y Caché en el frontend.
- `dashboard/app_evaluaciones.py`: Se enriqueció para mostrar las nuevas métricas derivadas de los archivos JSONL generados (TCO, latencia P95).

## 4. Documentación de Arquitectura y Evaluación
Archivos en `docs/arquitectura/` y `docs/evaluaciones/`:
- `estrategia_rag.md`: Se escribieron nuevas secciones dedicadas a explicar a detalle el *Semantic Cache*, el *Juez LLM* y los *Red-Teaming* ejecutados (System Override, Context Override).
- `arquitectura_rag.md`, `diagrama_arquitectura_evaluador.md` y `diagrama_arquitectura_nlp_core.md`: Se reescribió el código Mermaid (solucionando errores de sintaxis) para inyectar visualmente los nodos de Model Cascading y Caché Semántico.
- `guia_interpretacion_resultados.md`: Se agregó la sección 4 para que los usuarios puedan interpretar los nuevos cálculos (Correlación de Pearson, Proxy ECE, y Frontera de Pareto).

## 5. Cuadernos de Entrega y Raíz
- `notebooks/Avance5_#8.ipynb` (NUEVO): El Jupyter Notebook central donde se presenta de manera iterativa ("Hilo Conductor") toda la experimentación científica y técnica del Avance 5.
- `README.md`: Se actualizó el *Tech Stack* (añadiendo Streamlit) y se listaron las mejoras de la Fase 4 (Model Cascading).
- `requirements.txt`: Se hizo una limpieza, retirando librerías innecesarias (como `ragas`) y duplicados, formalizando explícitamente `numpy>=1.24.0` que se usaba por debajo para las operaciones de álgebra del evaluador.
- `.entregas.md`: Se actualizó localmente como System Prompt para indicar que en futuras entregas se debe aplicar la narrativa pedagógica implementada en este Avance 5.

### Conclusión del Avance
El objetivo principal del Avance 4 al 5 fue evolucionar el sistema de un estado de "investigación/baseline" a un ecosistema de grado productivo (MLOps), enfocándonos en auditoría experta, evaluación estadística estricta de la incertidumbre (Bootstrap/ECE), reducción dramática de latencia/costos (Model Cascading, Caché), y fortificación contra vulnerabilidades (Red-Teaming).
