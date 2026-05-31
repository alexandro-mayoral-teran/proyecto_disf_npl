import os

# --- 1. UPDATE estrategia_rag.md ---
filepath_rag = 'docs/arquitectura/estrategia_rag.md'
with open(filepath_rag, 'r', encoding='utf-8') as f:
    content_rag = f.read()

new_section_rag = """

## 10. Evolución Arquitectónica (Fase 4): Telemetría, Model Cascading y Análisis de Sesgos

A medida que el proyecto migró hacia métricas de producción, la arquitectura se ajustó para solucionar deficiencias empíricas y optimizar el Costo Total de Propiedad (TCO) y el rigor evaluativo:

### 10.1 Telemetría en Vivo y Dashboard Operativo
Se desarrolló un módulo de telemetría inyectado (`RastreadorTelemetria`) que intercepta las llamadas al LLM, midiendo latencia en milisegundos y tokens consumidos vía `tiktoken`. Esta información se persiste localmente en un formato de logs de alta eficiencia (`telemetria_llm.jsonl`) y se expone mediante un **Dashboard de Streamlit** (`dashboard/app_evaluaciones.py`). El dashboard permite al equipo monitorizar el TCO acumulado, los tiempos de inferencia y la distribución taxonómica de errores en tiempo real.

### 10.2 Abstracción de Modelos (Factory Pattern)
Se implementó `config_llm.py` usando el patrón *Factory*. A través de la variable de entorno `USE_LOCAL_LLM`, el orquestador conmuta dinámicamente entre el cliente de OpenAI (Nube) y los wrappers de Ollama (Local) sin refactorizar el código base.

### 10.3 Taxonomía de Errores (El fin del Blind Debugging)
Cuando el *LLM-as-a-Judge* dictamina que una extracción falló, un módulo subsecuente analiza el texto recuperado por ChromaDB y subdivide el fallo en:
*   **Error Tipo A (Retrieval):** El documento correcto nunca llegó al contexto. (Problema de Embeddings/Chunking).
*   **Error Tipo B (Generación/Leniency):** El documento sí llegó, pero el LLM lo ignoró, alucinó o falló su lógica matemática.
*   **Error Tipo C (Estructural):** Ruptura del contrato JSON (Pydantic).

### 10.4 Descubrimiento Empírico: LLM-as-a-Judge Leniency Bias
Durante las evaluaciones cruzadas de la Prueba Ciega (Data Contamination) y la Taxonomía, descubrimos un fenómeno documentado académicamente: **El sesgo de benevolencia**.
Cuando `llama3.1` (8B) actuó como juez evaluando sus propias respuestas, dictaminó 0 Errores B y aprobó como válidas el 37.6% de respuestas sin contexto (alucinadas). Sin embargo, cuando `gpt-4o` operó como juez sobre los mismos datos, fue implacable, detectando 43 Errores B y permitiendo solo un 5.5% de contaminación.

### 10.5 Arquitectura Definitiva: Model Cascading
Ante la evidencia matemática de la incapacidad del modelo pequeño para fungir como Juez de alto rigor, y el costo prohibitivo ($16 USD / 1k queries) de operar todo el flujo de extracción en `gpt-4o`, la arquitectura RAG adopta la estrategia de **Model Cascading (Enrutamiento Dinámico)**:
*   **Local (Llama 3.1):** Utilizado para tareas operativas de baja fricción, generación sintética base y re-ranking de búsquedas, manteniendo TCO=$0 y garantizando la privacidad de los datos extraídos (Residencia de Datos).
*   **Nube (GPT-4o-mini / GPT-4o):** Restringido exclusivamente como motor de Extracción Estructurada profunda y como **Juez Evaluador (Evaluator LLM)**, donde se requiere precisión quirúrgica en el seguimiento de esquemas Pydantic y detección de alucinaciones.
"""

if "10. Evolución Arquitectónica" not in content_rag:
    # Append to the end of the file
    content_rag += new_section_rag
    with open(filepath_rag, 'w', encoding='utf-8') as f:
        f.write(content_rag)
    print("Updated estrategia_rag.md")
else:
    print("estrategia_rag.md already has the section.")


# --- 2. UPDATE README.md ---
filepath_readme = 'README.md'
with open(filepath_readme, 'r', encoding='utf-8') as f:
    content_readme = f.read()

# Add Dashboard to bullet 5 in "4. Metodología de Evaluación"
old_bullet_5 = "5. **Telemetría y Costos (TCO):** Cada consulta registra latencia (P50/P95) y consumo de tokens, permitiendo cuantificar el costo por ejecución y analizar el espectro de **Vendor Lock-in** frente a modelos *Self-Hostable* (ej. Llama 3.1)."
new_bullet_5 = "5. **Telemetría Transparente y Dashboard (TCO):** Cada consulta registra latencia (P50/P95) y consumo de tokens usando `tiktoken`. Estos logs persisten en formato `.jsonl` y son monitorizados en tiempo real mediante un **Dashboard Interactivo en Streamlit**, permitiendo controlar el Gasto Operativo (OPEX) y contrastar la eficacia (Model Cascading) del ecosistema en Nube vs Local."

if old_bullet_5 in content_readme:
    content_readme = content_readme.replace(old_bullet_5, new_bullet_5)
    updated_readme = True
else:
    updated_readme = False
    print("Could not find bullet 5 in README.md")

# Add Model Cascading Architecture logic
old_arquitectura = "### 2. Pipeline Modular RAG (State of the Art)"
new_arquitectura = "### 2. Pipeline Modular RAG y Model Cascading\nLa arquitectura opera bajo el patrón de diseño **Factory** (`config_llm.py`), permitiendo conmutar al vuelo entre motores en Nube y Motores Locales para implementar **Model Cascading** (usar Llama 3.1 local para recuperación gratuita, y GPT-4o en Nube para síntesis estricta y evaluación)."

if old_arquitectura in content_readme:
    content_readme = content_readme.replace(old_arquitectura, new_arquitectura)
    updated_readme = True

if updated_readme:
    with open(filepath_readme, 'w', encoding='utf-8') as f:
        f.write(content_readme)
    print("Updated README.md")
