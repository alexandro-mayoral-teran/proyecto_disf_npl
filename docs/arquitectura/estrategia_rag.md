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

1. **Query Transformations (Multi-Query y HyDE):** Para cerrar la asimetría semántica entre consultas cortas de usuario y documentos legales extensos, se incorporan dos técnicas de expansión como "bloques LEGO":
   - **Multi-Query:** Genera 3 paráfrasis o perspectivas de la consulta original para cubrir distintos vocabularios.
   - **HyDE (Hypothetical Document Embeddings):** El LLM redacta una "alucinación" o documento hipotético respondiendo a la pregunta, imitando el tono y vocabulario oficial normativo. Luego, se vectoriza esta respuesta falsa para buscar coincidencias en ChromaDB (comparando documento vs documento). Esta técnica incrementa la capacidad de *Recall* en preguntas ambiguas.
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

### 3.6 Iteración del Sistema: Diagnóstico y Corrección de Pérdida de Contexto (Context Loss)

**El Ciclo de Iteración:**
1. **Creación de la Línea Base:** Inicialmente, se pobló la base de datos vectorial (ChromaDB) utilizando una estrategia estándar de fragmentación por encabezados (`MarkdownHeaderTextSplitter`).
2. **Evaluación Cuantitativa:** Al ejecutar el framework de evaluación (la "Arena de Modelos"), el sistema arrojó un *Recall* del 27.78%. 
3. **Diagnóstico del Problema:** El análisis de los errores en las evaluaciones reveló un problema estructural crítico: la **pérdida de contexto jerárquico**. En documentos complejos (como la CUB), una misma fórmula (ej. cálculo de la Probabilidad de Incumplimiento - PI) aparece múltiples veces, pero aplica a distintas carteras (revolvente, auto, nómina). Al fragmentar el texto, el *chunk* resultante contenía la fórmula pero quedaba "huérfano" perdiendo el contexto de la cartera a la que pertenecía (información que quedó "arriba" en el título). Por ello, consultas precisas como "PI de auto" fallaban.
4. **Corrección e Implementación (Post-procesamiento):** Para solucionar este cuello de botella y mejorar drásticamente las métricas, se iteró la arquitectura integrando soluciones directamente en `chunking.py` para inyectar este contexto antes de vectorizar.

**Soluciones Implementadas (Integradas como Post-procesadores en el Chunker):**
Para mantener una arquitectura limpia e integrada (alta cohesión), se implementó un patrón *Pipeline/Filtros*:

1. **Inyección de Metadatos (Metadata Injection):** Una técnica rápida y sin costo de API que concatena los títulos jerárquicos extraídos (ej. `[Contexto Estructural: Anexo 33 | Cartera Automotriz]`) físicamente al inicio del *chunk*. 
2. **Contextual Retrieval (Resumen con LLM Ligero - SOTA):** Una técnica de vanguardia donde, durante el indexado, se le envía el documento completo y el *chunk* a un LLM económico (como `gpt-4o-mini`). El LLM redacta una única oración de contexto (ej. *"Este fragmento detalla el cálculo de la PI específicamente para la cartera automotriz"*) que se antepone al texto. Esto elimina la ambigüedad semántica de forma definitiva, maximizando el *Recall* a cambio de costo y latencia moderados durante la ingesta.

**¿Por qué funciona esta concatenación física en el espacio vectorial?**
El modelo de embeddings (ej. OpenAI `text-embedding-3-small`) no lee ni interpreta los diccionarios de metadatos (JSON) de un objeto *Document*; procesa **exclusivamente el string de texto crudo (`page_content`)**. 
Si el texto fragmentado es simplemente *"La PI se calcula como X + Y"*, el vector resultante apunta semánticamente a conceptos matemáticos financieros, pero carece de la dimensión del tipo de producto. 
Al **concatenar físicamente** el contexto en un único "súper texto" (ej. `"[Contexto: Cartera Automotriz] La PI se calcula como X + Y"`), obligamos al modelo a "leer" todas las palabras juntas. El nuevo vector generado apunta simultáneamente al concepto matemático y al ecosistema automotriz. Cuando el usuario hace la consulta *"PI de auto"*, la similitud coseno hace un emparejamiento perfecto porque ambos vectores vibran en esa misma frecuencia semántica que originalmente se habría perdido.

**Compatibilidad entre Estrategias de Chunking y Post-procesadores:**
Dado que la arquitectura se diseñó bajo un modelo de componentes "LEGO" (Filtros), los post-procesadores nunca romperán el código independientemente de la estrategia de fragmentación seleccionada, pero su efectividad sí varía:
- **`ContextualizadorLLM`:** Es **100% universal**. Dado que el LLM lee el documento original completo y el fragmento resultante, es irrelevante si el fragmento se cortó por párrafos, por superposición fija (`fijo_overlap`) o por encabezados. El LLM siempre podrá inferir y redactar el contexto correctamente.
- **`InyectorMetadatos`:** Su efectividad **depende de la estrategia**. Esta clase busca activamente llaves como `"Header 1"` en el diccionario del chunk. Si se usa `EstrategiaChunking.ENCABEZADOS_MD`, LangChain extrae estos headers y funciona a la perfección. Sin embargo, si se usan cortadores básicos (`PARRAFO` o `FIJO_OVERLAP` sin encadenamiento previo), LangChain no extrae la jerarquía. En ese caso, el inyector no encontrará metadatos y dejará el *chunk* intacto (no inyectará contexto, pero el programa no fallará).

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

## 5. Resultados Evolutivos de la Arena de Modelos (Impacto del Contextualizador)

Se ejecutó un módulo de evaluación cuantitativa (Arena) midiendo la precisión de recuperación a través de tres escenarios evolutivos de indexación. Esto permitió medir empíricamente el valor añadido de las técnicas de mitigación de *Context Loss*. Los resultados a continuación utilizan el evaluador semántico **LLM-as-a-Judge (`llm_judge`)** para capturar aciertos parafraseados.

### 5.1 Escenario 1: Baseline (Only Chunking)
Los documentos se fragmentaron por encabezados sin post-procesamiento.
*   **Embeddings Puros:** Recall@10 del **60.0%**.
*   **MultiQuery + CrossEncoder:** Recall@10 del **60.0%**.
*   **Diagnóstico:** El modelo sufre "orfandad semántica". Fórmulas matemáticas profundas no pueden ser vinculadas al producto financiero al que pertenecen.

### 5.2 Escenario 2: Inyector de Metadatos
Se concatenó la ruta jerárquica (títulos y subtítulos) físicamente al inicio de cada *chunk* antes de vectorizar.
*   **Embeddings Puros:** Recall@10 sube a **66.67%**.
*   **MultiQuery + CrossEncoder:** Recall@10 sube a **70.0%**.
*   **Diagnóstico:** Efectividad comprobada. Obligar al modelo de embeddings a "leer" la jerarquía junto con el texto soluciona parcialmente la pérdida de contexto, mejorando directamente las métricas de recuperación.

### 5.3 Escenario 3: Contextual Retrieval (State of the Art)
Se utilizó `gpt-4o-mini` durante la ingesta para redactar un contexto explicativo holístico, anteponiéndolo al *chunk*. Se incorporó un pipeline adicional: *HyDE*.
*   **Embeddings Puros:** Recall@10 se dispara a **73.33%** (NDCG@10 de 0.7333).
*   **HyDE + Embeddings + CrossEncoder:** Alcanza **73.33%**.
*   **Diagnóstico:** El salto cualitativo es masivo. Se elimina la ambigüedad semántica por completo.

### 5.4 Escenario 4: El "Súper RAG" (Maximum Recall)
Se integró una configuración final combinando todas las técnicas en un único pipeline: Búsqueda Híbrida (RRF) expandida simultáneamente con Multi-Query y HyDE, filtrada por un Cross-Encoder.
*   **Híbrido + Ambos (Multi-Query/HyDE) + CrossEncoder:** El Recall@10 se disparó a un masivo **90.0%**.
*   **Diagnóstico:** Esta combinación representa el techo de cristal de la recuperación del sistema. Es extremadamente pesado computacionalmente (alta latencia), pero asegura que virtualmente ningún artículo regulatorio relevante quede fuera de la vista del modelo extractor.

### 5.5 Marco de Referencia (Benchmark de la Industria)
Para interpretar estos resultados, los estándares de la industria para RAG son:
*   **Recall@10:** 🔴 < 40% (Pobre), 🟡 40% - 60% (Aceptable), 🟢 60% - 85% (Bueno), ⭐ > 85% (SOTA con Fine-Tuning).
*   **NDCG@10:** 🔴 < 0.30 (Desordenado), 🟡 0.30 - 0.50 (Decente), 🟢 0.50 - 0.70 (Bueno a Excelente).

**Conclusión del Sistema Actual:** 
Haber escalado de un **60% a un 73.33%** posiciona al proyecto sólidamente en el rango **"Bueno a Excelente"** sin requerir *fine-tuning* de los embeddings, demostrando que el pre-procesamiento del contexto es más impactante que el modelo de vectorización per se.

### 5.6 Estrategia Seleccionada y Justificación Final
**La estrategia base de producción es: Embeddings Puros (con Contextual Retrieval). Sin embargo, el "Súper RAG" queda habilitado en la Interfaz Gráfica para consultas especializadas.**

**¿Por qué?**
1. **Precisión y UX Óptimas (Base):** Los Embeddings Puros alcanzan un excelente 73.33% con latencias de inferencia bajísimas (~0.55s - ~0.94s).
2. **Potencia a Demanda:** Mediante un Frontend visual y controles dinámicos (UI en Vanilla JS), el usuario puede activar la estrategia "Súper RAG" (90.0% Recall) cuando enfrenta requerimientos normativos complejos, aceptando conscientemente una latencia mayor (Cross-Encoder + Multi-expansion) a cambio de una precisión perfecta.

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

## 8. Roadmap y Próximos Pasos

El seguimiento detallado de tareas, hitos a futuro (Avance 5 - Ensambles y Avance 6 - Producción) y el registro formal de requerimientos científicos implementados se maneja en un documento vivo independiente. 

Para revisar el cronograma técnico actualizado y las rúbricas completadas, consulta el archivo: **[`support/pendientes_y_roadmap_final.txt`](../../support/pendientes_y_roadmap_final.txt)**

## 9. Justificación Técnica del Modelo Local: LLaMA 3.1 (8B)

En la arquitectura de nuestro pipeline RAG Híbrido, hemos seleccionado el modelo **LLaMA 3.1 de 8 mil millones de parámetros (`llama3.1:8b`)** de Meta para operar localmente a través de Ollama. 

A continuación, se documenta la justificación formal para esta decisión de diseño arquitectónico, evaluando sus ventajas, desventajas y comparativa con el mercado, con el fin de tener certidumbre técnica frente a los *sponsors* del proyecto.

### 9.1 ¿Por qué LLaMA 3.1 8B es la mejor opción actual? (Frontera de Pareto)

La elección de un modelo local para tareas NLP corporativas (como normativas de Banxico/DISF) se reduce a un problema de optimización entre **Hardware Requerido vs. Razonamiento Lógico**. LLaMA 3.1 8B se sitúa exactamente en el "Sweet Spot" de esta frontera:

**Ventajas Principales:**
1. **Ventana de Contexto Masiva (128k Tokens):** A diferencia de modelos anteriores que colapsaban a los 4k u 8k tokens, LLaMA 3.1 puede ingerir hasta 128,000 tokens. Esto es crítico para RAG.
2. **Hardware de Consumo:** Al estar cuantizado a 4-bits o 8-bits mediante Ollama, este modelo requiere entre 4.5 GB y 8 GB de RAM/VRAM para funcionar.
3. **Capacidad Multilingüe Mejorada:** Manejo del español normativo/legal excepcionalmente fluido.
4. **Instrucciones Estrictas:** Ajustado para seguir reglas duras (vital para HyDE y Pydantic).

**Desventajas:**
1. **Extracción Estructurada Profunda:** Aún se queda ligeramente atrás de modelos gigantes como GPT-4o en JSONs anidados complejos (por ello, delegamos esto a la nube).

### 9.2 Comparativa contra Alternativas (Por qué los descartamos)

- ❌ **Mistral v0.3 (7B) / Mixtral (8x7B):** Mistral 7B fue superado drásticamente por Llama 3.1. Mixtral requiere mucha más memoria (~24 GB de VRAM).
- ❌ **Gemma 2 (9B):** Excelente rendimiento, pero licencia comercial más restrictiva y mayor consumo computacional.
- ❌ **LLaMA 3.1 (70B):** Exige hardware empresarial de ultra-alta gama (Múltiples GPUs de 80GB).

### 9.3 Conclusión Contundente

> [!IMPORTANT]
> **Veredicto:** LLaMA 3.1 8B es la mejor opción estratégica actual para nuestro caso de uso prototipo (MVP). Nos provee un nivel de razonamiento a la par de GPT-3.5 de forma 100% privada.

### 9.4 Glosario Rápido
- **Cuantización:** Compresión de pesos matemáticos a 4/8 bits para ahorrar RAM.
- **Open-Weights:** Modelos descargables y ejecutables sin depender de APIs externas.

---

## 10. Viabilidad de Inferencia Local: Ollama vs vLLM en Hardware de Consumo

En esta sección se documenta el análisis técnico de viabilidad y los trade-offs de los motores de inferencia local en hardware de consumo típico de desarrollo.

### 10.1 Comparativa Técnica de Motores: Ollama vs vLLM

| Criterio | Ollama (Basado en `llama.cpp`) | vLLM (Inferencia de Producción) |
| :--- | :--- | :--- |
| **Arquitectura de Inferencia** | Inferencia síncrona/secuencial optimizada para CPU/GPU mixta. | Procesamiento por lotes continuo (*Continuous Batching*) y *PagedAttention*. |
| **Soporte de Hardware** | Multiplataforma (NVIDIA, AMD, Apple Silicon, CPU e Intel Arc). | Principalmente GPU NVIDIA dedicada (CUDA). Soporte experimental en AMD/ROCm. |
| **Instalación y Configuración** | Extremadamente sencilla. Un instalador nativo de un clic en Windows. | Compleja en Windows. Requiere instalar WSL2 (Linux), CUDA Toolkit y compilaciones C++. |
| **Gestión de Memoria** | Dinámica. Carga el modelo en VRAM y el excedente en RAM. Libera recursos al estar en reposo. | Estática. Se adueña del 90% (por defecto) de la VRAM desde la inicialización, independientemente del uso. |
| **Rendimiento (Latency / Throughput)** | Excelente latencia para un usuario (~45-55 tokens/s con RTX 4060 en 8B). | Altísimo throughput bajo alta concurrencia (cientos de consultas simultáneas). |

### 10.2 Ventajas y Desventajas en Entornos de Desarrollo Local

#### Ollama (La Opción Elegida para Desarrollo)
*   **Ventajas:** Consumo bajo demanda y estabilidad nativa en Windows sin virtualización pesada.
*   **Desventajas:** No optimizado para cientos de usuarios paralelos.

#### vLLM (La Opción Recomendada para Producción)
*   **Ventajas:** Máximo rendimiento concurrente (SOTA).
*   **Desventajas e Impacto en Laptops:** Congela la máquina al adueñarse de la VRAM. Fricción de WSL2/Linux.

### 10.3 Justificación de la Estrategia Híbrida de Inferencia
Para el proyecto DISF, se adopta un **Enfoque de Inferencia Evolutivo**:
1.  **Fase de Desarrollo y Prototipado (Local):** Se utiliza **Ollama** con el modelo `llama3.1:8b`. 
2.  **Fase de Despliegue en Producción:** Se migrará a **vLLM** alojado en un contenedor Docker en un servidor dedicado (ej. AWS/GCP con L4). Es 100% compatible con OpenAI API.

## 11. Estrategias de Ensambles y Calibración (Avance 5 - ENS)

Para maximizar la robustez del sistema y cumplir con las rúbricas avanzadas, el proyecto adopta una estrategia de **Ensambles Heterogéneos**.

1. **El Ensamble de Recuperación (RRF):** En lugar de confiar en un solo modelo de *retrieval*, el sistema ensambla modelos con sesgos inductivos opuestos: un modelo estadístico léxico (`TfidfVectorizer` / `BM25`) y un modelo semántico profundo (`text-embedding-3-small`). Estos "Base Learners" se fusionan mediante *Reciprocal Rank Fusion (RRF)*, logrando un ensamble robusto que supera consistentemente a sus partes individuales.
2. **Diversidad Cuantificada (ENS-D):** Para garantizar que el ensamble realmente aporta valor, se evalúa empíricamente la tasa de desacuerdo (*Disagreement Rate*) entre los modelos individuales. Si ambos modelos fallan en los mismos casos (Correlación de errores > 0.8), el ensamble es redundante.
3. **Calibración y Consistencia (ENS-E):** En entornos regulatorios, la certeza es primordial. Se empleará la técnica de *Self-Consistency* (ejecutar la misma consulta múltiples veces con `temperature > 0`) para medir si las respuestas del LLM se mantienen estables o si divergen ante el mismo contexto, lo que indicaría miscalibración o "alucinación latente".

## 12. Arquitectura de Producción y Seguridad Cloud (Avance 6 - DEP)

La transición de un entorno de evaluación (Eval) a un sistema productivo exige rigurosos controles operacionales y de seguridad.

1. **Costo Total de Propiedad a 12 Meses (TCO - DEP-B):** La viabilidad financiera se sustenta comparando el TCO de la solución Self-hosted (Hardware + Electricidad + vLLM) frente al esquema Multi-Cloud (Costo por 1000 tokens en OpenAI). Se mapea explícitamente el almacenamiento vectorial (*Vector Index Storage*) y la retención de logs.
2. **SLOs y Monitoreo en Producción (DEP-C):**
   - **Latencia:** Se establece un Service Level Objective numérico (ej. $P_{95} \le 3.5$ segundos).
   - **Drift Detection:** Monitoreo activo de la distribución de consultas entrantes. Si las consultas de los usuarios divergen radicalmente de nuestro Eval Set congelado, se disparan alertas de retuning.
3. **Seguridad y Red-Teaming (DEP-D):**
   - **Prompt Injection & Jailbreaks:** Pruebas documentadas intentando "romper" los guardrails del Agente (ej. forzar respuestas destructivas).
   - **Manejo de PII:** Dado que es un sistema para analistas bancarios, la inferencia local (Llama 3) funge como escudo primario para consultas sensibles, aislando los datos de la nube pública.
4. **Plan de Handoff (DEP-E):** Entrega documentada de artefactos serializados (Base vectorial indexada en ChromaDB, Registro de Prompts en JSON), garantizando que el sponsor institucional pueda operar, detener o destruir limpiamente los activos del proyecto (*Decommissioning plan*).


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
