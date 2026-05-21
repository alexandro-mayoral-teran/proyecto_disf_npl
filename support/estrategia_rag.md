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
2. **Vectorización e Indexación (`src/nlp_core/vectorizacion.py`):** Los fragmentos generados se convierten en vectores matemáticos usando `text-embedding-3-small` de OpenAI (un modelo optimizado y económico) y se almacenan localmente utilizando **ChromaDB**. Adicionalmente, durante las etapas de evaluación se utiliza `TfidfVectorizer` (haciendo explícita la normalización L2 con `norm='l2'`, fundamental para estandarizar el cálculo de similitud coseno) para establecer el marco de referencia léxico y capturar la jerga financiera. Como mejora a futuro se explorará el *fine-tuning* contrastivo de embeddings sobre pares (regulación, campo_formulario) del dominio Banxico para una mejor representación.
3. **Consulta y Extracción Híbrida (Hybrid Search):** El proyecto evolucionó de una simple búsqueda semántica a un modelo de *Reciprocal Rank Fusion (RRF)*. Ahora, cuando se solicita información, el sistema ejecuta dos búsquedas en paralelo:
   *   **Búsqueda Semántica (ChromaDB + Embeddings):** Recupera fragmentos conceptualmente relevantes, incluso si usan sinónimos o lenguaje indirecto.
   *   **Búsqueda Léxica Exacta (BM25):** Recupera fragmentos donde las palabras clave (como un número de artículo o un código de formato específico) coinciden exactamente.
   
   **Mecanismo de Fusión (RRF):** Dado que es matemáticamente inconsistente promediar una distancia geométrica de coseno (Embeddings) con una ponderación estadística de frecuencias (BM25), el sistema implementa el algoritmo *Reciprocal Rank Fusion*. Este método ignora los *scores* absolutos originales y recalcula la relevancia basándose estrictamente en la posición de llegada (el *ranking*) que cada fragmento obtuvo en ambas listas. El resultado es una lista maestra ordenada de forma democrática. Los artículos resultantes se entregan al Agente, mitigando tanto la "deriva semántica" de los vectores como la rigidez de las palabras clave.

### C. Arquitectura Map-Reduce
Aunque el RAG híbrido es eficiente, corre un "riesgo de omisión" si la búsqueda vectorial falla en traer un artículo crítico. La solución planteada para estos casos es un patrón *Map-Reduce*: recuperar **todos** los artículos de un anexo específico mediante metadatos, aplicar el agente extractor a cada uno de manera individual (Map), y utilizar un agente consolidador final para unir los resultados en el esquema definitivo (Reduce).

### D. Pipeline Modular de RAG Avanzado (SOTA - Avance 3)
Para maximizar la precisión y superar consistentemente la línea base (BM25), se implementa un pipeline de recuperación con técnicas avanzadas orientadas a cerrar la brecha entre consultas cortas y documentos regulatorios largos:

1. **Query Transformations (HyDE y Query Expansion):** Implementación de *Hypothetical Document Embeddings* (HyDE) generando 2-3 documentos hipotéticos por consulta mediante el LLM y promediando sus embeddings. Asimismo, se emplea expansión de consultas utilizando sinónimos regulatorios extraídos del vocabulario TF-IDF para potenciar drásticamente el *recall*.
2. **Cross-Encoder Reranking Jerárquico:** Tras la Búsqueda Híbrida (RRF), se introduce una segunda etapa indispensable utilizando un modelo especializado (ej. `sentence-transformers/mmarco-MiniLMv2-L12-H384` en español). Este modelo lee simultáneamente la pregunta y el documento, reordenando el *Top K* final y compensando la "deriva semántica" inherente a los embeddings. Opcionalmente, se aplicará un *LLM-as-judge* para validación contextual.
3. **Context Compression (Compresión de Contexto):** Extracción exclusiva de las oraciones relevantes dentro de los fragmentos recuperados. Esto reduce el ruido (*Lost in the Middle*) y minimiza el uso de tokens.

### E. Framework de Evaluación Cuantitativa (Validación Avance 3)
Un componente esencial es la evaluación sistemática y cuantitativa del pipeline RAG:

1. **Dataset de Evaluación (Ground Truth):** Como paso inicial, se construye un dataset benchmark de 20-30 consultas representativas, cada una emparejada con 3-5 documentos relevantes (anotados por expertos de Banxico). Esta base hace falsable el experimento.
2. **Métricas IR Estándar y Métrica Primaria:** Se comparará cuantitativamente (Recall@5, Recall@10, MAP@10, NDCG@10) entre enfoques BoW, TF-IDF, Embeddings simples, Híbrida y Híbrida+Reranking. La métrica primaria a priori se establece como **Recall@5** (o alternativamente **nDCG@10**).
3. **Prueba de Contaminación de Datos (Data Contamination Check):** Evaluación del LLM sin contexto inyectado (*no-context test*) para discriminar entre la capacidad real del pipeline RAG y la potencial memorización de la normativa durante el pre-entrenamiento del LLM.
4. **Análisis de Errores por Etapas:** Clasificación estructurada de los errores del sistema en tres categorías: (a) El sistema de Retrieval falló; (b) Retrieval exitoso pero el LLM alucinó; (c) Retrieval y respuesta exitosos pero formato estructurado inválido.
5. **ROI y Costo de Revisión Humana:** Análisis comparativo del tiempo (minutos/documento) que requiere un analista de la DISF para validar las salidas del sistema versus el proceso de análisis manual, justificando económicamente el uso de IA.

## 4. Estado Actual (Roadmap)

1.  ✅ **Ingesta y Limpieza:** Flujo completado. Refactorizado bajo la clase unificada `IngestorDocumentos`, logrando pasar de PDF crudo (vía BlazeDocs), Excel (.xlsx) y Word (.docx) a Markdown limpio de forma transparente.
2.  ✅ **Desarrollo del Chunking (OOP):** Implementado con el patrón de diseño *Strategy* (`RegulacionChunker`), lo que permite cambiar fácilmente entre estrategias jerárquicas (Markdown) o de tamaño fijo.
3.  ✅ **Vectorización y Retrieval (Desacoplados):** Arquitectura separada limpiamente en `MotorVectorizacion` (para poblar ChromaDB) y `MotorBusqueda` (para consultas y recuperación).
4.  ✅ **Búsqueda Híbrida Implementada:** Integración de la librería `rank_bm25` y `EnsembleRetriever` de LangChain para fusionar búsquedas semánticas y por palabras clave.
5.  ✅ **Adaptación del Agente:** El agente Pydantic es capaz de manejar los esquemas estrictos de extracción, evaluando *Full Context* vs *RAG Simple*.
6.  🔄 **Evaluación Comparativa (La "Arena"):** El proyecto se encuentra actualmente construyendo el Notebook de benchmarking final para comparar la latencia, el costo en tokens y la precisión (frente a un Golden Dataset) de los enfoques *Full Context*, *TF-IDF* y *Búsqueda Híbrida*.

## 5. Horizonte de Producción (Escalabilidad)

Inspirado en arquitecturas de despliegue real, se contemplan las siguientes integraciones para la fase de exposición vía API:

1. **Caché Semántico:** Implementación de una capa de memoria que almacene respuestas previas. Ante consultas semánticamente idénticas, el sistema devolverá el resultado cacheado, mitigando costos de API y reduciendo la latencia a milisegundos.
2. **Operaciones de Índice (Index Ops):** Módulo CRUD dedicado para ChromaDB. Facilitará la actualización quirúrgica de la base vectorial (ej. re-vectorizar únicamente los artículos que sufran reformas) sin necesidad de reprocesar el corpus íntegro.
3. **Control de Acceso (RBAC):** Utilización estricta de metadatos en la recuperación vectorial para restringir la inyección de contexto normativo según el nivel de autorización o perfil del analista dentro de la DISF.
