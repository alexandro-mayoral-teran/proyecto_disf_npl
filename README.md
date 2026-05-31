# 🏦 Diseño Automatizado de Requerimientos de Información Financiera con NLP

**Proyecto Integrador de Maestría**  
**Dominio:** Banco de México - Dirección de Información del Sistema Financiero (DISF)

---

## 📌 Descripción Ejecutiva del Proyecto

### Contexto y Problemática
De acuerdo con la Ley del Banco de México, una de las funciones prioritarias de la institución es procurar la estabilidad y el sano desarrollo del sistema financiero. Para lograrlo, la Dirección de Información del Sistema Financiero (DISF) recaba y analiza información mediante "requerimientos de información" (formularios y catálogos) diseñados con base en vasta y compleja documentación regulatoria (ej. la Circular Única de Bancos).

Actualmente, el proceso de traducir disposiciones normativas (información no estructurada en PDFs y documentos extensos con cientos de páginas, fórmulas matemáticas y reglas de negocio) en estructuras de datos tabulares es una tarea manual, altamente especializada e intensiva en tiempo, propensa a inconsistencias técnicas y operativas.

### Propuesta de Solución
Este proyecto automatiza esta extracción mediante una herramienta basada en **Procesamiento de Lenguaje Natural (NLP)** y **Modelos de Lenguaje Grande (LLMs)**. El sistema actúa como un **"Especialista Digital Regulador"**, capaz de ingerir documentos legales complejos e identificar automáticamente variables de información relevantes.

Su núcleo funcional se basa en un esquema estricto de validación de arquitecturas de datos (mediante `Pydantic`) para asegurar que el modelo estructure el texto en el formato exacto requerido por los sistemas institucionales de bases de datos del Banco (ej. SAIFWeb).

### Valor Institucional
Esta iniciativa representa un primer paso hacia la adopción de Inteligencia Artificial para la automatización de procesos críticos de supervisión. La herramienta **no busca reemplazar el juicio del experto financiero, sino empoderarlo**, reduciendo drásticamente el tiempo de "traducción" de requerimientos y asegurando la estandarización y calidad de los datos acopiados por el Banco Central.

---

## 🏗️ Estrategias de IA y Arquitectura del Sistema

El proyecto sigue una arquitectura modular "API-First" implementando técnicas de vanguardia en Generación Aumentada por Recuperación (RAG) específicas para textos regulatorios.

### 1. Ingesta y Limpieza de Datos
*   **Ingesta con BlazeDocs (`src/ingesta/`):** Se transforman los documentos normativos a un formato Markdown (`.md`) estructural. Esto preserva la jerarquía semántica (títulos, subtítulos) y convierte las tablas visuales en texto legible por máquina.
*   **Limpieza de Ruido Institucional:** Uso de expresiones regulares para eliminar firmas, marcas de agua del "Diario Oficial", y numeración de páginas sin alterar la estructura Markdown.

### 2. Pipeline Modular RAG y Model Cascading
La arquitectura opera bajo el patrón de diseño **Factory** (`config_llm.py`), permitiendo conmutar al vuelo entre motores en Nube y Motores Locales para implementar **Model Cascading** (usar Llama 3.1 local para recuperación gratuita, y GPT-4o en Nube para síntesis estricta y evaluación).
*   **Fragmentación Estructural (Chunking):** Se utiliza `MarkdownHeaderTextSplitter` para fragmentar manteniendo la coherencia jerárquica (Títulos, Capítulos, Artículos).
*   **Diagnóstico y Mitigación de Pérdida de Contexto:** Se implementaron inyectores de metadatos y **Contextual Retrieval** con LLMs ligeros (`gpt-4o-mini`) para anteponer un resumen de contexto antes de vectorizar el *chunk*, resolviendo la "orfandad" semántica de fragmentos profundos (como fórmulas matemáticas específicas en incisos).
*   **Consulta y Extracción Híbrida (Hybrid Search RRF):** Búsqueda en paralelo mediante semántica pura (ChromaDB + OpenAI Embeddings) y léxica exacta (BM25), uniendo resultados bajo *Reciprocal Rank Fusion (RRF)* para capturar la terminología y jerga financiera exacta de forma democrática.
*   **Query Transformations:** Implementación de *Multi-Query* (paráfrasis de consulta para cubrir distintos vocabularios) y *HyDE* (Hypothetical Document Embeddings) para expandir y conectar consultas cortas con textos normativos densos.
*   **Cross-Encoder Reranking Jerárquico:** Paso post-recuperación que utiliza un modelo especializado que lee simultáneamente la pregunta y el documento para reordenar el Top K final.

### 3. Normalización y Telemetría
*   **Base Matemática:** Aplicación rigurosa de Normalización L2 explícita para equiparar la Similitud Coseno al Producto Punto (acelerando ChromaDB). Se descartó intencionalmente la estandarización clásica (como `StandardScaler`) post-vectorización para no saturar memoria ni destruir los pesos originales de la Frecuencia Inversa de Documento (IDF).
*   **Telemetría Transversal:** Rastreo detallado de latencia (en segundos) y consumo preciso de tokens en el contexto utilizando `tiktoken`.

---

## 📊 Metodología de Evaluación (Arquitectura Config-Driven)

Nuestro marco de evaluación ha evolucionado hacia un modelo científico y riguroso, alineado a estándares MLOps, preparado para calcular la Frontera de Pareto y garantizar resultados estadísticamente significativos:

1. **Dataset Expandido (Golden Dataset):** El conjunto de validación creció a **109 consultas representativas** y balanceadas por complejidad, longitud y tipo de documento normativo.
2. **Matriz de Pruebas Dinámica:** Se configuraron 6 estrategias de RAG independientes (desde baselines léxicos hasta Súper RAG con Expansión y Re-ranking), orquestadas directamente desde `config_experimentos.json` sin tocar código fuente.
3. **Validación Estadística (Bootstrap CI):** Implementación de remuestreo Bootstrap (1,000 iteraciones) para generar Intervalos de Confianza al 95% sobre el *NDCG@10*, detectando empates estadísticos o superioridad real entre la Nube y ejecución Local.
4. **Desagregación de Errores:** Se implementó una Prueba Ciega (Data Contamination) y una taxonomía automática de fallos para identificar cuellos de botella exactos:
   - **Fallo Tipo A:** Error en la Recuperación (Retrieval no encontró el texto).
   - **Fallo Tipo B:** Alucinación Generativa (El LLM falló a pesar de tener contexto).
   - **Fallo Tipo C:** Error Estructural (Fallo al generar el JSON validado por Pydantic).
5. **Telemetría Transparente y Dashboard (TCO):** Cada consulta registra latencia (P50/P95) y consumo de tokens usando `tiktoken`. Estos logs persisten en formato `.jsonl` y son monitorizados en tiempo real mediante un **Dashboard Interactivo en Streamlit**, permitiendo controlar el Gasto Operativo (OPEX) y contrastar la eficacia (Model Cascading) del ecosistema en Nube vs Local.

---

## 🎯 Alcance y Entregables (Producto Mínimo Viable)

El MVP extrae y genera los siguientes artefactos:
1. **Estructura tabular de los formularios:** Nombres de campo, tipos de dato, longitudes y descripciones funcionales.
2. **Propuesta de catálogos asociados:** Identificación automática de variables que requieren listas cerradas de valores.
3. **Auditoría y Mejoras:** Detección de ambigüedades en el texto legal y sugerencias proactivas de validaciones de negocio.

---

## 📚 Documentación Oficial

El proyecto cuenta con una estructura formal de documentación técnica alojada en la carpeta `docs/` para guiar a analistas e ingenieros.

### 🧪 Evaluaciones y Pruebas
*   [Manual de Ejecución de Pruebas](docs/evaluaciones/manual_ejecucion_pruebas.md): Guía paso a paso para configurar y correr el pipeline de evaluación (Config-Driven).
*   [Guía de Interpretación de Resultados](docs/evaluaciones/guia_interpretacion_resultados.md): Cómo leer y analizar las métricas (Recall, NDCG, Data Contamination y Error Breakdown).

### 🏛️ Arquitectura y Modelos
*   [Arquitectura RAG (Hub de Diagramas)](docs/arquitectura/arquitectura_rag.md): Visión general de la arquitectura y punto de entrada a diagramas especializados.
*   [Estrategia RAG y Justificación de Modelos](docs/arquitectura/estrategia_rag.md): Documento maestro sobre búsqueda híbrida (RRF), re-ranking, control de alucinaciones y la justificación formal del uso de Llama 3.1 8B vs Nube.
*   **Diagramas Especializados (Mermaid):**
    *   [Arquitectura Interna del NLP Core](docs/arquitectura/diagrama_arquitectura_nlp_core.md)
    *   [Pipeline del Módulo de Ingesta](docs/arquitectura/diagrama_arquitectura_ingesta.md)
    *   [Arquitectura del Juez Evaluador](docs/arquitectura/diagrama_arquitectura_evaluador.md)

### 💻 Setup y Entorno Local
*   [Manual LLM Local](docs/setup/manual_llm_local.md): Configuración de los modelos open-source mediante Ollama y vLLM.

---

## 🚀 Mapa de Ruta y Próximos Pasos

| Fase | Estado | Descripción Técnica |
| :--- | :--- | :--- |
| **1. Cimentación y Dataset** | 🟢 Completado | Extracción a Markdown estructural y limpieza. Refactorización a Arquitectura OOP. |
| **2. RAG Engine y Retrieval** | 🟢 Completado | Chunking jerárquico, Búsqueda Híbrida (RRF), Contextual Retrieval y Query Transformations. |
| **3. Evaluación Cuantitativa** | 🟢 Completado | Construcción de Arena con 8 Pipelines, telemetría de latencia/tokens y Ground Truth. |
| **4. Evaluación y Selección (Avance 4)** | 🟢 Completado | Frontera de Pareto, Bootstrapping (95% CI), Taxonomía de Errores (A/B/C), Contaminación Ciega (Lift). Dataset de 110 consultas. |
| **5. Ensambles y Calibración (Avance 5)** | 🟡 En Progreso | **Siguientes pasos:** Cuantificar diversidad del ensamble RRF, calibrar salidas probabilísticas y justificar la Frontera de Pareto final. |
| **6. Producción y Seguridad (Avance 6)** | ⚪ Pendiente | TCO a 12 meses, SLOs, Pruebas de Seguridad (Red-Teaming) y Plan de Handoff para el Banco de México. |
| **7. Interfaz y Despliegue (MVP Final)** | 🟢 Completado | Despliegue mediante **FastAPI** y una interfaz interactiva (**Vanilla JS + Flexbox**) estilo Banxico, con telemetría RAG en vivo. |
---

## 🛠️ Stack Tecnológico

*   **Lenguaje Principal:** Python 3.10+
*   **Procesamiento y Telemetría:** Pandas, Tabulate, Expresiones Regulares (`re`), `tiktoken`.
*   **Contrato de Datos:** Pydantic.
*   **Vectorización y Retrieval (RAG):** ChromaDB, BM25 (`rank_bm25`), OpenAI Embeddings (`text-embedding-3-small`), Modelos Cross-Encoder.
*   **Generación LLM:** OpenAI (`gpt-4o`, `gpt-4o-mini`) y modelos locales Open-Source vía Ollama o vLLM (conmutación transparente mediante configuración en `.env`).
*   **Despliegue Web:** FastAPI (Motor Backend) y Vanilla JS / CSS (Frontend Interactivo).

## ⚙️ Configuración del Entorno (Desarrollo Local)

1. Clonar el repositorio.
2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instalar dependencias *(Archivo requirements.txt en construcción)*:
   ```bash
   pip install -r requirements.txt
   ```