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

### 3.1 Full Context Prompting (La Línea Base)
*   **En qué consiste:** Se inyecta el documento normativo completo en el prompt del LLM. Se utiliza un riguroso *Prompt Engineering*, instruyendo al modelo a actuar como un "Especialista Digital Regulador" y exigiéndole aplicar reglas estrictas (ej. *"extrae fórmulas"*, *"crea catálogos si ves listas cerradas"*, *"reporta ambigüedades"*).
*   **Ventajas:** El modelo tiene todo el contexto simultáneamente, permitiéndole correlacionar una regla en el Capítulo I con una excepción en el Capítulo V.
*   **Desventajas:** Un costo financiero muy elevado en consumo de tokens. Además, sufre del fenómeno cognitivo *"Lost in the Middle"*, donde el modelo tiende a ignorar instrucciones si la ventana de contexto se vuelve demasiado grande.

### 3.2 Generación Aumentada por Recuperación (RAG)
Para mitigar el alto costo y evitar la pérdida de contexto, se diseñó un flujo RAG especializado en el ámbito regulatorio:

1. **Fragmentación Estructural / Chunking (`src/nlp_core/chunking.py`):** En textos normativos, cortar cada "N" tokens es peligroso porque puede partir un artículo por la mitad. Se implementó `MarkdownHeaderTextSplitter` para fragmentar el documento respetando la jerarquía (Títulos, Capítulos, Artículos). Esto asegura que un artículo y su tabla mantengan coherencia, inyectando la "ruta del documento" como un *metadato* del fragmento.
2. **Vectorización e Indexación (`src/nlp_core/vectorizacion.py`):** Los fragmentos generados se convierten en vectores matemáticos usando `text-embedding-3-small` de OpenAI (un modelo optimizado y económico) y se almacenan localmente utilizando **ChromaDB**. Adicionalmente, durante las etapas de evaluación se utiliza `TfidfVectorizer` para establecer el marco de referencia léxico y capturar la jerga financiera.
3. **Consulta y Extracción Híbrida (Hybrid Search):** El proyecto evolucionó de una simple búsqueda semántica a un modelo de *Reciprocal Rank Fusion (RRF)*. Ahora, cuando se solicita información, el sistema ejecuta dos búsquedas en paralelo:
   *   **Búsqueda Semántica (ChromaDB + Embeddings):** Recupera fragmentos conceptualmente relevantes, incluso si usan sinónimos o lenguaje indirecto.
   *   **Búsqueda Léxica Exacta (BM25):** Recupera fragmentos donde las palabras clave coinciden exactamente.
   *   **Mecanismo de Fusión (RRF):** Este método ignora los *scores* absolutos originales y recalcula la relevancia basándose estrictamente en la posición de llegada (el *ranking*) que cada fragmento obtuvo en ambas listas. El resultado es una lista maestra ordenada de forma democrática.

### 3.3 Arquitectura Map-Reduce
Aunque el RAG híbrido es eficiente, corre un "riesgo de omisión" si la búsqueda vectorial falla en traer un artículo crítico. La solución planteada para estos casos es un patrón *Map-Reduce*: recuperar **todos** los artículos de un anexo específico mediante metadatos, aplicar el agente extractor a cada uno de manera individual (Map), y utilizar un agente consolidador final para unir los resultados en el esquema definitivo (Reduce).

### 3.4 Pipeline Modular de RAG Avanzado (SOTA)
Para maximizar la precisión y superar consistentemente la línea base (BM25), se implementa un pipeline de recuperación con técnicas avanzadas orientadas a cerrar la brecha entre consultas cortas y documentos regulatorios largos:

1. **Query Transformations (Multi-Query y Expansión):** Se emplea expansión de consultas generando 2-3 variantes por consulta mediante el LLM (ChatGPT) para potenciar drásticamente el *recall*.
2. **Cross-Encoder Reranking Jerárquico:** Tras la Búsqueda Híbrida (RRF), se introduce una segunda etapa utilizando un modelo especializado. Este modelo lee simultáneamente la pregunta y el documento, reordenando el *Top K* final y compensando la "deriva semántica" inherente a los embeddings.

### 3.5 Aspectos Matemáticos y Telemetría: Normalización L2 y Estandarización
Para construir nuestra línea base léxica (TF-IDF) y semántica (Embeddings) de manera rigurosa, se documentan decisiones matemáticas y métricas clave aplicadas en el código:

1. **Normalización L2 Explícita para TF-IDF y Semántica:** 
   - **En Léxico (TF-IDF):** Hemos declarado explícitamente `norm='l2'` en la instanciación de `TfidfVectorizer`. 
   - **En Semántica (OpenAI Embeddings):** Los modelos `text-embedding-3-small` ya emiten vectores pre-normalizados en L2 de forma nativa. 
   La normalización L2 ajusta matemáticamente cada vector para que su magnitud sea exactamente 1. El impacto es absoluto: **la Similitud Coseno se vuelve idéntica a un simple Producto Punto (`Inner Product`)**. Esto permite a bases de datos vectoriales como ChromaDB acelerar inmensamente los cálculos en producción (optimizaciones SIMD) y utilizar umbrales de relevancia universales independientemente del largo del texto.
   
2. **Estandarización Post-Vectorización (Ausencia Intencional):**
   Es una mala práctica aplicar una estandarización clásica (como `StandardScaler`, que resta la media y divide por la desviación estándar) después de vectorizar. 
   - **Pérdida de Esparsidad:** Restar la media convierte una matriz "dispersa" en una densa, saturando la RAM.
   - **Destrucción de Pesos Originales:** La varianza unitaria trata a todas las palabras o dimensiones latentes como si tuvieran la misma importancia, destruyendo el propósito principal de la Frecuencia Inversa de Documento (IDF).

3. **Telemetría Transversal (Tokens y Latencia):**
   Se implementó seguimiento robusto durante el retrieval contando tokens precisos inyectados en el contexto (mediante `tiktoken`) y trackeando la latencia entre etapas (Cross-Encoder, OpenAI calls para expansión).

## 4. Framework de Evaluación Cuantitativa

Un componente esencial es la evaluación sistemática y cuantitativa del pipeline RAG:

### 4.1 Dataset de Evaluación y Métricas IR
1. **Dataset de Evaluación (Ground Truth):** Se construyó un dataset benchmark emparejando consultas representativas con documentos relevantes esperados.
2. **Diccionario de Métricas:**
   - **Recall@K:** Mide si el documento relevante apareció entre los primeros $K$ resultados. Si es bajo, ChatGPT alucinará por falta de contexto.
   - **MAP@K (Mean Average Precision):** Penaliza drásticamente a los motores que "entierran" la respuesta correcta al fondo.
   - **NDCG@K (Normalized Discounted Cumulative Gain):** El *Estándar de Oro*. Usa atenuación logarítmica premiando colocar la respuesta en el Rank #1. Nuestra métrica directriz para ajustes será el **NDCG@10**.

### 4.2 Métodos de Evaluación del Hit en el Retrieval
Evaluar si un documento recuperado es un "acierto" se diseñó utilizando un patrón polimórfico en el `EvaluadorRAG`, permitiendo tres modos de ejecución:

1. **A. Subcadena Exacta (`exact_match`):** Busca si el texto clave esperado está contenido físicamente en el fragmento recuperado. Es ultrarrápido y no consume tokens de API. Sirve como la línea base estricta, aunque penaliza falsamente documentos con respuestas parafraseadas o redundancias en la ley.
2. **B. Revisión Manual (`human` y cálculo posterior):** 
   - **Exportación:** El sistema recupera el Top K y exporta una plantilla de Excel (`auditoria_manual_Estrategia.xlsx`) con las consultas y fragmentos para que un experto califique manualmente (1 o 0).
   - **Cálculo:** Una vez que el analista devuelve el archivo calificado, el evaluador cuenta con el método `calcular_metricas_desde_excel()` que parsea las filas, agrupa por consulta, extrae el Rank más alto calificado con '1', y calcula automáticamente el Recall, MAP y NDCG con la máxima precisión matemática basada en criterio experto ("Human-in-the-loop").
3. **C. LLM como Juez (`llm_judge` - Context Relevance):** Un modelo avanzado (`gpt-4o-mini`) actúa como juez imparcial leyendo la pregunta y el contexto recuperado. Evalúa semánticamente si el fragmento contiene la información necesaria y emite un fallo binario al vuelo. Resuelve la limitante del `exact_match` de manera automatizada.

### 4.3 Otros Controles de Calidad
1. **Prueba de Contaminación de Datos:** Evaluación del LLM sin contexto inyectado (*no-context test*) para medir la memorización nativa.
2. **Análisis de Errores por Etapas:** Clasificar si (a) El Retrieval falló; (b) Retrieval exitoso pero LLM alucinó; (c) Formato JSON inválido.

## 5. Resultados del Baseline (Arena de Modelos)

Se ejecutó un módulo de evaluación cuantitativa (Arena) enfrentando 8 pipelines distintos. 

### 5.1 Resultados Finales de la Arena (Telemetría y Precisión)

| Estrategia | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Latencia Prom. (s) | Tokens Contexto |
|:---|---:|---:|---:|---:|---:|---:|
| 1. BoW | 22.22% | 22.22% | 0.2222 | 0.2222 | 0.021 | 117.9 |
| 2. TF-IDF | 22.22% | 22.22% | 0.2222 | 0.2222 | 0.015 | 117.9 |
| 3. Híbrido BoW+TFIDF | 22.22% | 22.22% | 0.2222 | 0.2222 | 0.282 | 117.9 |
| 4. Embeddings | 27.78% | 27.78% | 0.2778 | 0.2778 | 0.412 | 131.3 |
| 5. Híbrido RRF | 27.78% | 27.78% | 0.2778 | 0.2778 | 0.591 | 131.3 |
| 6. Híbrido + CrossEncoder | 27.78% | 27.78% | 0.2778 | 0.2778 | 7.345 | 131.3 |
| 7. Embeddings + CrossEncoder | 27.78% | 27.78% | 0.2778 | 0.2778 | 6.458 | 131.3 |
| 8. MultiQuery + Emb + CrossEncoder | 27.78% | 27.78% | 0.2778 | 0.2778 | 8.874 | 131.3 |

**Hallazgos Clave:**
1. **La Semántica domina sobre lo Léxico:** El salto de 22.22% a 27.78% lo dan los Embeddings. Buscar por coincidencias exactas es ineficiente en jerga regulatoria.
2. **El costo del Cross-Encoder:** Aumentó el costo de inferencia brutalmente (de ~0.4s a ~6.4s) sin lograr una ganancia palpable de métricas en este set específico.
3. **El Costo del Multi-Query:** Introdujo latencia adicional (de 6.4s a 8.8s) por los llamados al LLM sin mejoras directas sobre el Retrieval basal en este ground truth determinista.

### 5.2 Marco de Referencia (Benchmark de la Industria)

Para interpretar estos resultados, los estándares de la industria para RAG son:
*   **Recall@10:** 🔴 < 40% (Pobre), 🟡 40% - 60% (Aceptable), 🟢 60% - 85% (Bueno), ⭐ > 85% (Estado del Arte con Fine-Tuning).
*   **NDCG@10:** 🔴 < 0.30 (Desordenado), 🟡 0.30 - 0.50 (Decente), 🟢 0.50 - 0.70 (Bueno a Excelente).

### 5.3 Diagnóstico del Sistema Actual
Nuestro sistema base se sitúa en un **Recall del 27.78%**. Aunque es bajo según el benchmark teórico, representa un **éxito de ingeniería** por:
1. **Rigor del Ground Truth:** La validación determinista estricta penaliza falsamente (con un 0) los aciertos semánticos o parafraseos si no contienen el ID exacto.
2. **Jerga Financiera:** Los embeddings generales (`text-embedding-3-small`) sufren con terminología como "ACT_i".
3. **Línea Base Cuantitativa:** Hemos expuesto matemáticamente los límites del modelo *Out-of-the-Box*. Sabemos que para llegar a un Recall >70% requeriremos afinar *chunks*, probar `text-embedding-3-large`, o aplicar *Fine-Tuning*.

### 5.4 Estrategia Seleccionada y Justificación Final
**La estrategia elegida para el producto final en producción es: Híbrido RRF (Estrategia 5) o Embeddings Puros (Estrategia 4).**

**¿Por qué?**
1. **Precisión sin Sacrificar UX:** Alcanzan el máximo rendimiento de Recall (27.78%) y NDCG con latencias bajísimas (~0.4s a ~0.6s).
2. **Telemetría:** Consumen aproximadamente ~131 tokens de contexto por recuperación, siendo ultra-eficientes económicamente.
3. **Escalabilidad:** Las arquitecturas complejas (Cross-Encoder y Multi-Query) quedan programadas como "piezas de lego" opcionales por si la complejidad de las consultas futuras lo amerita.

## 6. Estado Actual (Roadmap)

1.  ✅ **Ingesta y Limpieza:** Flujo completado (`IngestorDocumentos`).
2.  ✅ **Desarrollo del Chunking (OOP):** Patrón *Strategy* (`RegulacionChunker`) implementado.
3.  ✅ **Vectorización y Retrieval (Desacoplados):** Arquitectura separada limpiamente en `MotorVectorizacion` y `MotorBusqueda`.
4.  ✅ **Búsqueda Híbrida Implementada:** Integración de `rank_bm25` y `EnsembleRetriever`.
5.  ✅ **Adaptación del Agente:** Agente Pydantic funcional.
6.  ✅ **Pipeline Modular de Retrieval (LEGO Style):** Pipeline robusto soportando Expansión (`Multi-Query`), BM25, Embeddings y Re-ranking.
7.  ✅ **Framework de Evaluación Avanzado:** Implementación de métricas IR usando tres modos dinámicos (Subcadena, Juez LLM y Revisión Humana offline en Excel).
8.  ✅ **Telemetría y Evaluación Comparativa (Arena finalizada):** Benchmark ejecutado midiendo latencia, tokens y métricas IR.

## 7. Horizonte de Producción (Escalabilidad)

Inspirado en arquitecturas de despliegue real, se contemplan las siguientes integraciones:
1. **Caché Semántico:** Memoria para almacenar respuestas previas y reducir costos de API y latencia a milisegundos en consultas repetidas.
2. **Operaciones de Índice (Index Ops):** Módulo CRUD para ChromaDB, permitiendo actualizar únicamente los artículos que sufran reformas sin reprocesar todo el corpus.
3. **Control de Acceso (RBAC):** Utilización de metadatos en ChromaDB para restringir la inyección de contexto según el perfil del analista dentro de la DISF.
