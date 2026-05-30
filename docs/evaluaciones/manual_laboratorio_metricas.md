# 📊 Manual del Laboratorio de Métricas Avanzadas (E5/E6)

Este documento oficializa la arquitectura de evaluación de segunda generación construida para la **Dirección de Información del Sistema Financiero (DISF)** de Banco de México, diseñada para cumplir con el máximo rigor científico exigido en las rúbricas de evaluación de LLMs (Avances 5 y 6).

---

## 1. Arquitectura del Laboratorio

El laboratorio está compuesto por dos dominios principales, separados físicamente de la aplicación en producción (`app/`):

1.  **Motor Matemático de Backend:** Ubicado en `src/lab/`, orquesta la ejecución de pruebas contra el LLM de forma programática.
2.  **Frontend Analítico:** Ubicado en `dashboard/app_evaluaciones.py`, utiliza `Streamlit` para ingestar los CSVs generados y renderizar gráficas interactivas para la toma de decisiones.

---

## 2. Dimensiones Evaluadas (Backend)

Se implementaron tres módulos analíticos especializados para expandir la simple medición de "precisión" (NDCG) hacia dominios de robustez empresarial:

### 2.1. Diversidad Cuantificada (`src/lab/diversidad_eval.py`) - Rúbrica ENS-D
Mide matemáticamente la discrepancia entre dos modelos (Ej. Llama 3.1 Local vs GPT-4o Nube).
*   **Métricas Clave:** 
    *   *Disagreement Rate* (Porcentaje de veces que uno acierta y el otro falla).
    *   *Matriz de Acuerdo* (Matriz de Confusión cruzada).
    *   *Oracle Gap* (El "Lift" máximo teórico si lográramos ensamblar ambos modelos perfectamente).

### 2.2. Calibración y Consistencia (`src/lab/consistencia_eval.py`) - Rúbrica ENS-E
Evalúa el "Self-consistency" (Confianza) del modelo.
*   **Métrica Clave:** *Paraphrase Invariance Score*. Ejecuta la misma consulta N veces (con temperatura > 0) y utiliza un Juez LLM para verificar si las respuestas mantienen el mismo valor de verdad semántico. Si el score es bajo, el modelo está miscalibrado o propenso a alucinar.

### 2.3. Seguridad y Red-Teaming (`src/lab/seguridad_eval.py`) - Rúbrica DEP-D
Prueba de penetración automatizada contra los "Guardrails" del sistema.
*   **Métrica Clave:** *Tasa de Defensa*. Lee un Golden Dataset venenoso (`data/01_raw/eval_dataset_red_teaming.json`) con 10 ataques categorizados (Prompt Injection, Extracción de PII, Jailbreaks) y verifica que el sistema bloquee el vector de ataque.

---

## 3. Uso del Dashboard Analítico (Frontend)

El Dashboard es la herramienta principal para que los Data Scientists e Ingenieros validen los resultados. 

**Para inicializar el entorno:**
```bash
streamlit run dashboard/app_evaluaciones.py
```

### Funcionalidades por Pestaña:
*   **Frontera de Pareto (MA7/ENS-C):** Scatter plot interactivo que cruza la precisión (NDCG@10) contra el costo operativo por consulta. Permite justificar matemáticamente la elección del modelo final (Trade-off Costo vs Precisión).
*   **Taxonomía Automática de Errores (MA6):** Gráficas de pastel basadas en pruebas ciegas (Data Contamination) que categorizan los fallos en:
    *   **A:** Fallo de Recuperación (Retrieval).
    *   **B:** Alucinación del Modelo.
    *   **C:** Fallo Estructural de Pydantic.
*   **Diversidad y Seguridad:** Renderiza las métricas generadas por `diversidad_eval.py` y el reporte JSON de Red-Teaming de `seguridad_eval.py`.

---

> **Nota para Evaluadores y Stakeholders:**
> Toda métrica producida por estos módulos cuenta con significancia estadística respaldada por el algoritmo de Remuestreo Bootstrap CI (1,000 iteraciones) documentado en el *Manual de Ejecución de Pruebas*.
