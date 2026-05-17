# Estrategias de Procesamiento de Lenguaje Natural para el Proyecto DISF

Este documento sirve como la documentación extendida del proyecto. Detalla el contexto del problema, el flujo de procesamiento de los documentos regulatorios, las técnicas de Prompt Engineering aplicadas y las estrategias de Inteligencia Artificial evaluadas (Full Context vs RAG) para la extracción automatizada de requerimientos de información.

## 1. Contexto del Problema

El Banco de México (Banxico) y otras entidades regulatorias publican normativas extensas (ej. Circular Única de Bancos). Estas normativas contienen reglas de negocio, fórmulas matemáticas y catálogos de datos que las instituciones financieras deben cumplir al reportar su información. 

El reto principal radica en que **esta información no está estructurada**. Extraer manualmente qué campos, validaciones y catálogos componen un "Formulario de Reporte" a partir de cientos de páginas de texto legal es un proceso lento, costoso y propenso a errores humanos. El objetivo del proyecto DISF es automatizar esta extracción, convirtiendo texto legal denso en esquemas de datos estructurados (JSON / Pydantic) listos para implementarse en bases de datos institucionales.

## 2. Flujo de Preparación de Documentos (Ingesta y Limpieza)

Antes de que cualquier modelo de IA pueda interpretar la normativa, los documentos originales (comúnmente PDFs) deben ser procesados con altísima fidelidad:

1. **Ingesta con BlazeDocs (`src/ingesta/`):** Los documentos legales suelen tener formatos complejos (múltiples columnas, tablas insertadas, pies de página). Se utiliza la API de BlazeDocs para convertirlos en archivos Markdown (`.md`). Esto preserva la jerarquía semántica (títulos, subtítulos) y convierte las tablas visuales en tablas de texto legibles por la máquina.
2. **Limpieza de Ruido (`src/utils/limpieza_texto.py`):** Los documentos extraídos contienen "ruido" originado por el OCR y la paginación (ej. marcas de agua del "Diario Oficial de la Federación", avisos de derechos reservados, firmas, números de página). El script aplica expresiones regulares para eliminar este ruido institucional sin alterar la estructura Markdown, garantizando un flujo de lectura continuo para el modelo.

## 3. Estrategias de Extracción e IA Aplicadas

Para transformar el texto limpio en un formulario estructurado, el proyecto emplea un Agente basado en `gpt-4o` acoplado a un esquema estricto de validación usando Pydantic (`RequerimientoInformacion`). Se desarrollaron y documentaron múltiples aproximaciones arquitectónicas (visibles en `src/nlp_core/agente.py`):

### A. Full Context Prompting (La Línea Base)
*   **En qué consiste:** Se inyecta el documento normativo completo en el prompt del LLM. Se utiliza un riguroso *Prompt Engineering*, instruyendo al modelo a actuar como un "Especialista Digital Regulador" y exigiéndole aplicar reglas estrictas (ej. *"extrae fórmulas"*, *"crea catálogos si ves listas cerradas"*, *"reporta ambigüedades"*).
*   **Ventajas:** El modelo tiene todo el contexto simultáneamente, permitiéndole correlacionar una regla en el Capítulo I con una excepción en el Capítulo V.
*   **Desventajas:** Un costo financiero muy elevado en consumo de tokens. Además, sufre del fenómeno cognitivo *"Lost in the Middle"*, donde el modelo tiende a ignorar instrucciones si la ventana de contexto se vuelve demasiado grande.

### B. Generación Aumentada por Recuperación (RAG)
Para mitigar el alto costo y evitar la pérdida de contexto, se diseñó un flujo RAG especializado en el ámbito regulatorio:

1. **Fragmentación Estructural / Chunking (`src/nlp_core/chunking.py`):** En textos normativos, cortar cada "N" tokens es peligroso porque puede partir un artículo por la mitad. Se implementó `MarkdownHeaderTextSplitter` para fragmentar el documento respetando la jerarquía (Títulos, Capítulos, Artículos). Esto asegura que un artículo y su tabla mantengan coherencia, inyectando la "ruta del documento" como un *metadato* del fragmento.
2. **Vectorización e Indexación (`src/nlp_core/vectorizacion.py`):** Los fragmentos generados se convierten en vectores matemáticos usando `text-embedding-3-small` de OpenAI (un modelo optimizado y económico) y se almacenan localmente utilizando **ChromaDB**.
3. **Consulta y Extracción Híbrida (Hybrid Search):** El proyecto evolucionó de una simple búsqueda semántica a un modelo de *Reciprocal Rank Fusion (RRF)*. Ahora, cuando se solicita información, el sistema ejecuta dos búsquedas en paralelo:
   *   **Búsqueda Semántica (ChromaDB + Embeddings):** Recupera fragmentos conceptualmente relevantes, incluso si usan sinónimos o lenguaje indirecto.
   *   **Búsqueda Léxica Exacta (BM25):** Recupera fragmentos donde las palabras clave (como un número de artículo o un código de formato específico) coinciden exactamente.
   
   **Mecanismo de Fusión (RRF):** Dado que es matemáticamente inconsistente promediar una distancia geométrica de coseno (Embeddings) con una ponderación estadística de frecuencias (BM25), el sistema implementa el algoritmo *Reciprocal Rank Fusion*. Este método ignora los *scores* absolutos originales y recalcula la relevancia basándose estrictamente en la posición de llegada (el *ranking*) que cada fragmento obtuvo en ambas listas. El resultado es una lista maestra ordenada de forma democrática. Los artículos resultantes se entregan al Agente, mitigando tanto la "deriva semántica" de los vectores como la rigidez de las palabras clave.

### C. Arquitectura Map-Reduce
Aunque el RAG híbrido es eficiente, corre un "riesgo de omisión" si la búsqueda vectorial falla en traer un artículo crítico. La solución planteada para estos casos es un patrón *Map-Reduce*: recuperar **todos** los artículos de un anexo específico mediante metadatos, aplicar el agente extractor a cada uno de manera individual (Map), y utilizar un agente consolidador final para unir los resultados en el esquema definitivo (Reduce).

### D. Pipeline Modular de RAG Avanzado (SOTA - En desarrollo)
Para maximizar la precisión en escenarios regulatorios complejos sin comprometer innecesariamente la latencia o los costos computacionales, se está implementando un pipeline de recuperación altamente modular. Dado que no todas las consultas requieren la máxima potencia algorítmica, el `MotorBusqueda` permitirá habilitar o deshabilitar dinámicamente las siguientes técnicas del estado del arte:

1. **Multi-Query Expansion:** Utilización del LLM para generar reformulaciones sintácticas de la consulta original. Esto permite lanzar múltiples búsquedas en paralelo, mitigando el problema de la "brecha de vocabulario".
2. **Cross-Encoder Reranking:** Tras realizar la Búsqueda Híbrida (RRF), los candidatos resultantes pasan por un modelo especializado (ej. *sentence-transformers*) que lee simultáneamente la pregunta y el documento para emitir una calificación de relevancia de alta precisión, reordenando el *Top K* definitivo.
3. **Context Compression (Compresión de Contexto):** Aplicación de algoritmos para extraer exclusivamente las oraciones relevantes de los chunks recuperados. Esto reduce el ruido semántico (*Lost in the Middle*) y minimiza radicalmente el consumo de tokens.

## 4. Estado Actual (Roadmap)

1.  ✅ **Ingesta y Limpieza:** Flujo completado. Refactorizado bajo la clase unificada `IngestorDocumentos`, logrando pasar de PDF crudo (vía BlazeDocs), Excel (.xlsx) y Word (.docx) a Markdown limpio de forma transparente.
2.  ✅ **Desarrollo del Chunking (OOP):** Implementado con el patrón de diseño *Strategy* (`RegulacionChunker`), lo que permite cambiar fácilmente entre estrategias jerárquicas (Markdown) o de tamaño fijo.
3.  ✅ **Vectorización y Retrieval (Desacoplados):** Arquitectura separada limpiamente en `MotorVectorizacion` (para poblar ChromaDB) y `MotorBusqueda` (para consultas y recuperación).
4.  ✅ **Búsqueda Híbrida Implementada:** Integración de la librería `rank_bm25` y `EnsembleRetriever` de LangChain para fusionar búsquedas semánticas y por palabras clave.
5.  ✅ **Adaptación del Agente:** El agente Pydantic es capaz de manejar los esquemas estrictos de extracción, evaluando *Full Context* vs *RAG Simple*.
6.  🔄 **Evaluación Comparativa (La "Arena"):** El proyecto se encuentra actualmente construyendo el Notebook de benchmarking final para comparar la latencia, el costo en tokens y la precisión (frente a un Golden Dataset) de los enfoques *Full Context*, *TF-IDF* y *Búsqueda Híbrida*.
