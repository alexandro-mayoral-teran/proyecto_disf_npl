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

### 2. Pipeline Modular RAG (State of the Art)
*   **Fragmentación Estructural (Chunking):** Se utiliza `MarkdownHeaderTextSplitter` para fragmentar manteniendo la coherencia jerárquica (Títulos, Capítulos, Artículos).
*   **Diagnóstico y Mitigación de Pérdida de Contexto:** Se implementaron inyectores de metadatos y **Contextual Retrieval** con LLMs ligeros (`gpt-4o-mini`) para anteponer un resumen de contexto antes de vectorizar el *chunk*, resolviendo la "orfandad" semántica de fragmentos profundos (como fórmulas matemáticas específicas en incisos).
*   **Consulta y Extracción Híbrida (Hybrid Search RRF):** Búsqueda en paralelo mediante semántica pura (ChromaDB + OpenAI Embeddings) y léxica exacta (BM25), uniendo resultados bajo *Reciprocal Rank Fusion (RRF)* para capturar la terminología y jerga financiera exacta de forma democrática.
*   **Query Transformations:** Implementación de *Multi-Query* (paráfrasis de consulta para cubrir distintos vocabularios) y *HyDE* (Hypothetical Document Embeddings) para expandir y conectar consultas cortas con textos normativos densos.
*   **Cross-Encoder Reranking Jerárquico:** Paso post-recuperación que utiliza un modelo especializado que lee simultáneamente la pregunta y el documento para reordenar el Top K final.

### 3. Normalización y Telemetría
*   **Base Matemática:** Aplicación rigurosa de Normalización L2 explícita para equiparar la Similitud Coseno al Producto Punto (acelerando ChromaDB). Se descartó intencionalmente la estandarización clásica (como `StandardScaler`) post-vectorización para no saturar memoria ni destruir los pesos originales de la Frecuencia Inversa de Documento (IDF).
*   **Telemetría Transversal:** Rastreo detallado de latencia (en segundos) y consumo preciso de tokens en el contexto utilizando `tiktoken`.

---

## 📊 Metodología de Evaluación y Resultados (Arena de Modelos)

Se empleó un marco riguroso de evaluación cuantitativa basado en un **Golden Dataset de IR (Ground Truth)**. El evaluador implementa un patrón polimórfico con 3 modos: **Subcadena Exacta (`exact_match`)**, **Juez LLM (`llm_judge` - Context Relevance)**, y **Revisión Manual (Human-in-the-loop mediante plantillas exportadas en Excel)**.

### Resultados de la Arena 
El desempeño del motor de búsqueda se evaluó bajo 3 escenarios evolutivos para mitigar la "pérdida de contexto", utilizando un juez LLM imparcial:
1. **Only Chunking (Línea base):** Alcanzó un **60.0%** de *Recall@10* con Embeddings puros, sufriendo de "orfandad semántica".
2. **Inyector de Metadatos:** Al concatenar la ruta jerárquica físicamente al chunk, el *Recall@10* subió al **66.67%**.
3. **Contextual Retrieval (SOTA):** Usando un LLM en tiempo de ingesta para resumir y anteponer el contexto global al chunk, el *Recall@10* alcanzó un impresionante **73.33%**.
4. **Súper RAG (Híbrido + Multi-Query/HyDE + Cross-Encoder):** La combinación exhaustiva de técnicas de expansión, fusión de rangos y re-ordenamiento elevó el *Recall@10* hasta un **90.0%**, estableciendo el límite superior de precisión del sistema.

**Conclusión Técnica:** La estrategia seleccionada para producción base es **Embeddings Puros (con Contextual Retrieval)** por su balance (73.33% con ~0.55s de latencia). Sin embargo, el **Súper RAG** (90.0%) se mantiene configurado como el motor de "Alta Precisión" disponible a través de la Interfaz Visual, asumiendo una latencia mayor por el uso del *Cross-Encoder* y expansión doble.

---

## 🎯 Alcance y Entregables (Producto Mínimo Viable)

El MVP extrae y genera los siguientes artefactos:
1. **Estructura tabular de los formularios:** Nombres de campo, tipos de dato, longitudes y descripciones funcionales.
2. **Propuesta de catálogos asociados:** Identificación automática de variables que requieren listas cerradas de valores.
3. **Auditoría y Mejoras:** Detección de ambigüedades en el texto legal y sugerencias proactivas de validaciones de negocio.

---

## 🚀 Mapa de Ruta y Próximos Pasos

| Fase | Estado | Descripción Técnica |
| :--- | :--- | :--- |
| **1. Cimentación y Dataset** | 🟢 Completado | Extracción a Markdown estructural y limpieza. Refactorización a Arquitectura OOP. |
| **2. RAG Engine y Retrieval** | 🟢 Completado | Chunking jerárquico, Búsqueda Híbrida (RRF), Contextual Retrieval y Query Transformations. |
| **3. Evaluación Cuantitativa** | 🟢 Completado | Construcción de Arena con 8 Pipelines, telemetría de latencia/tokens y Ground Truth. |
| **4. Selección de Modelos (Avance 4)** | ⚪ Pendiente | **Siguientes pasos científicos:** Integración de LLMs Open-Source (Llama 3/Mistral vía Ollama) para residencia de datos local en el Banco. Pruebas Ciega (Data Contamination), Frontera de Pareto (Costo vs Precisión), Análisis Desagregado de Errores (Retrieval vs Alucinación) e Intervalos de Confianza. |
| **5. Horizonte de Producción (API y UI)** | 🟢 Completado | Despliegue mediante **FastAPI** y una interfaz interactiva (**Vanilla JS + Flexbox**) estilo Banxico, con telemetría RAG en vivo y selectores dinámicos de pipeline. |

---

## 🛠️ Stack Tecnológico

*   **Lenguaje Principal:** Python 3.10+
*   **Procesamiento y Telemetría:** Pandas, Tabulate, Expresiones Regulares (`re`), `tiktoken`.
*   **Contrato de Datos:** Pydantic.
*   **Vectorización y Retrieval (RAG):** ChromaDB, BM25 (`rank_bm25`), OpenAI Embeddings (`text-embedding-3-small`), Modelos Cross-Encoder.
*   **Generación LLM:** OpenAI (`gpt-4o`, `gpt-4o-mini`). Próxima adaptación a Ollama / vLLM.
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