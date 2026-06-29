# Estrategia RAG y Arquitectura del Proyecto ARIF

Este documento sirve como la documentación extendida del proyecto. Detalla el contexto del problema, el flujo de procesamiento de los documentos regulatorios, las técnicas de Prompt Engineering aplicadas y las estrategias de Inteligencia Artificial evaluadas (Full Context vs RAG) para la extracción automatizada de requerimientos de información.

> **Nota de Arquitectura Actualizada:** Para conocer los detalles operativos de la extracción estructurada final (Long-Context + Pydantic) y la inyección automatizada de metadatos de gobernanza que se implementaron como prototipos finales en el sistema, consulta el documento unificado: [extraccion_y_gobernanza_datos.md](./extraccion_y_gobernanza_datos.md).

## 1. Introducción y Contexto del Problema

El Banco de México (Banxico) y otras entidades regulatorias publican normativas extensas (ej. Circular Única de Bancos). Estas normativas contienen reglas de negocio, fórmulas matemáticas y catálogos de datos que las instituciones financieras deben cumplir al reportar su información. 

El reto operativo principal radica en tres factores:
1. **Interpretar** esta normatividad requiere de un alto nivel de especialización y conocimiento experto.
2. **Consumo de tiempo:** Leer y analizar la regulación con el nivel de detalle necesario es sumamente lento.
3. **Falta de Estandarización:** Al intervenir diferentes equipos, los formularios de requerimiento diseñados carecen de estructura uniforme.

Extraer manualmente qué campos, validaciones y catálogos componen un "Formulario de Reporte" a partir de cientos de páginas de texto legal es un proceso propenso a errores. El objetivo del proyecto ARIF (Asistente Regulatorio para Información Financiera) para la DISF es automatizar esta extracción, convirtiendo texto legal denso en esquemas de datos estructurados (JSON / Pydantic) listos para implementarse en bases de datos institucionales.

## 2. Evolución del Proyecto y Fundamentos Arquitectónicos

Inicialmente, el proyecto buscaba saltar directamente a un asistente que diseñara formularios de manera automatizada. Sin embargo, hubo un punto de inflexión crítico: **antes de que la IA pueda "diseñar", primero necesita "entender" la regulación sin alucinar**. Por ello, el cimiento indispensable es un **Sistema RAG (Retrieval-Augmented Generation)** altamente preciso.

### 2.1 Ingesta Multi-Formato y Limpieza Institucional

Antes de que cualquier modelo de IA pueda interpretar la normativa, los documentos originales deben ser procesados con altísima fidelidad:

1. **Ingesta Multi-Formato y Justificación de Markdown (`src/ingesta/ingestor.py`):** Los documentos legales oficiales suelen publicarse en formatos complejos como PDF. Aunque el PDF es excelente visualmente, es ineficiente para las máquinas porque la paginación rompe la lectura continua (tablas y párrafos partidos a la mitad). Para solucionarlo, se utiliza la API de BlazeDocs para PDFs y librerías nativas para procesar documentos de Word (`.docx`) y tablas en Excel (`.xlsx`), convirtiéndolos unificadamente a texto plano en formato **Markdown (`.md`)**. 
   * **¿Por qué en markdown?** Al convertir a Markdown, estructuramos explícitamente el documento usando sintaxis de encabezados (`# Título`, `## Capítulo`, `### Artículo`). Esto es el habilitador tecnológico que permite posteriormente fragmentar el texto respetando la jerarquía de la regulación, de modo que el sistema RAG sepa exactamente a qué Título y Capítulo pertenece cualquier párrafo, mejorando radicalmente la precisión de recuperación (Retrieval) y evitando la "orfandad semántica".
2. **Resiliencia en la Extracción Documental (Fallback Local PyPDF):** Si BlazeDocs falla (ej. por archivos corruptos como "Git LFS Pointers"), el flujo se rutea silenciosamente hacia un parser local (`pypdf`). Si el archivo también falla localmente, el sistema emite un HTTP 400 amistoso informando al usuario que el archivo original está corrupto, preservando la UX sin crashear el orquestador.
3. **Limpieza de Ruido Institucional (`src/utils/limpieza_texto.py`):** Se desarrollaron expresiones regulares para limpiar ruidos originados por el OCR (CNBV, BANXICO, DOF, BASEL). Esto remueve pies de página, encabezados, fechas y firmas sin alterar la estructura normativa, garantizando un flujo de lectura continuo.

### 2.2 Fragmentación Estructural y Vectorización Matemática

1. **Estrategias de Fragmentación / Chunking (`src/nlp_core/chunking.py`):** El particionado de textos legales es una decisión arquitectónica crítica. Se evaluaron múltiples estrategias para procesar los documentos y evitar la pérdida de sentido:
   * **Naive Chunking:** La opción básica de cortar el texto cada "N" cantidad de tokens fijos fue descartada rápidamente. En textos normativos, este enfoque ciego partía artículos o tablas a la mitad, destruyendo por completo el contexto legal y matemático.
   * **Fragmentación Estructural (Implementado):** Se optó por utilizar `MarkdownHeaderTextSplitter`. Esta estrategia fragmenta el documento respetando la jerarquía natural de la ley (Títulos, Capítulos, Artículos) generada durante la conversión a Markdown. Esto asegura que la lógica jurídica se mantenga intacta, inyectando la "ruta del documento" como un *metadato* del fragmento.
   * **Parent-Child Chunking (Evaluado):** Consiste en dividir el documento en fragmentos "Padres" extensos y subdividirlos en "Hijos" cortos para ChromaDB. Aunque ofrece alta precisión de búsqueda, se descartó temporalmente porque requeriría modificar profundamente la clase `MotorBusqueda` para triangular los IDs.
   * **RAPTOR / Árboles Recursivos (Evaluado):** Agrupa y resume recursivamente los chunks en un árbol semántico. Es ideal para responder preguntas globales ("Resume todo el manual"), pero se determinó que añade demasiada complejidad a la ingesta y riesgo de duplicar respuestas en el prompt.
   * **Contextual Retrieval Asíncrono (Seleccionado como SOTA):** Ante las limitantes anteriores, se determinó que esta es la estrategia más robusta y transparente para el pipeline de Retrieval actual. Utiliza un LLM durante la ingesta para inyectar el contexto global al chunk aislado. Para mitigar el masivo cuello de botella en tiempo (que tomaba horas de forma secuencial), se reescribió la implementación utilizando procesamiento asíncrono en Python (`asyncio.gather`), reduciendo la vectorización a escasos minutos.
2. **Vectorización e Indexación (`src/nlp_core/vectorizacion.py`):** Los fragmentos de texto procesados se transforman en representaciones matemáticas (vectores de 1536 dimensiones) utilizando el modelo `text-embedding-3-small` de OpenAI, elegido por su alta eficiencia y excelente captura de semántica en español jurídico a bajo costo. 
   * **Implementación:** Durante la indexación, el código inyecta metadatos enriquecidos (institución, tema, versión, documento origen) a cada vector para permitir filtrado granular posterior. Además, se generan IDs deterministas (ej. `f"{ruta_archivo.stem}_{chunker.estrategia}_chunk_{i}"`) garantizando la idempotencia del proceso, lo que permite actualizar documentos en la base de datos sin duplicar registros.
   * **Base de Datos Vectorial (ChromaDB vs Alternativas):** Los vectores se almacenan localmente en **ChromaDB**.
     * *ChromaDB (Implementado):* Se eligió por ser open-source, ejecutar localmente apoyado en disco (SQLite/Parquet) y no requerir infraestructura externa. Es ideal para nuestro enfoque "Local-First" y prototipado ágil. Su contraparte es que no escala distributivamente out-of-the-box para millones de transacciones concurrentes.
     * *Pinecone:* Plataforma SaaS totalmente gestionada. Ideal para producción masiva y latencia ultrabaja, pero introduce dependencia de un proveedor (vendor lock-in) y costos recurrentes, rompiendo el esquema 100% local.
     * *Weaviate / Milvus:* Bases de datos vectoriales open-source de grado empresarial. Excelentes para búsquedas híbridas nativas a escala y despliegues en Kubernetes (K8s), pero añaden una alta carga de ingeniería y mantenimiento operativo (DevOps) que no era justificable para el volumen documental actual del proyecto ARIF.
3. **Normalización L2 Explícita para TF-IDF y Semántica:** Se declaró `norm='l2'` en `TfidfVectorizer` y los embeddings de OpenAI ya vienen pre-normalizados. Esto ajusta cada vector para que su magnitud sea exactamente 1, convirtiendo la Similitud Coseno en un simple Producto Punto (`Inner Product`), acelerando inmensamente los cálculos en ChromaDB. Evitamos intencionalmente la Estandarización Clásica para no perder esparsidad.
4. **Consulta y Extracción Híbrida (Hybrid Search):** Para combatir la vulnerabilidad de depender de un solo motor de búsqueda, el sistema ejecuta dos algoritmos ortogonales en paralelo (`src/nlp_core/retrieval.py`):
   *   **Semántica (Embeddings en ChromaDB):** Es altamente resistente a errores tipográficos y sinónimos. Excelente para recuperar lenguaje indirecto o conceptos amplios. Sin embargo, suele fallar estrepitosamente en recuperar identificadores precisos (ej. "Artículo 42-Bis" o acrónimos).
   *   **Léxica Exacta (BM25 en RAM):** Algoritmo estadístico basado en la frecuencia de términos. Es la red de seguridad perfecta para recuperar nombres propios, acrónimos y números de artículos exactos, cubriendo el punto ciego de los Embeddings.
   *   **Fusión Matemática (RRF):** Dado que ChromaDB devuelve distancias (Coseno) y BM25 devuelve puntajes asimétricos (Okapi), sus métricas base no son matemáticamente sumables. Se implementó el algoritmo *Reciprocal Rank Fusion* (RRF) con una constante `k=60`. Este método ignora el score absoluto y usa únicamente la posición de llegada de cada motor (`1 / (rank + k)`) para recalcular una lista maestra verdaderamente democrática.
5. **Query Transformations (Transformación de Consultas):** Para lidiar con la barrera de lenguaje entre el usuario y el regulador, las peticiones se interceptan antes de tocar la base de datos:
   *   **Multi-Query Expansion:** Un usuario podría buscar "préstamos de autos", pero la ley dictamina "cartera automotriz". Un LLM rápido genera 3 paráfrasis ortogonales de la consulta original para cubrir la inmensa varianza del vocabulario humano. Cada paráfrasis lanza un hilo de búsqueda independiente, los cuales se re-ensamblan posteriormente eliminando duplicados.
   *   **HyDE (Hypothetical Document Embeddings):** En lugar de vectorizar la corta pregunta del usuario, se fuerza al LLM a "alucinar" o redactar una respuesta hipotética asumiendo el tono normativo de la institución. Esa respuesta falsa pero extensa es la que se vectoriza. Esto transforma un problema de búsqueda asimétrico (pregunta corta vs documento largo) en uno simétrico (documento vs documento), incrementando drásticamente las coincidencias en el espacio vectorial y el *Recall*.

## 3. Hallazgos Evaluativos y la Solución Definitiva (El "Súper RAG")

### 3.1 Framework de Evaluación y Hallazgos Empíricos

Se construyó una "Arena de Evaluación" con un Golden Dataset de ~100 consultas reales sobre liquidez y crédito. 

**Métricas de Recuperación de Información (IR):**
1. **Recall@K:**
   * *Concepto:* Mide el porcentaje de veces que el fragmento con la respuesta correcta apareció dentro de los primeros `K` resultados devueltos (ej. Recall@10).
   * *Cálculo:* `(Número de documentos relevantes recuperados en el top K) / (Total de documentos relevantes esperados)`. Al buscar una única respuesta correcta, suele evaluarse de forma binaria: 1 si aparece en el Top K, 0 si no.
   * *Pros:* Es la métrica de supervivencia en RAG. Si el Recall es 0% (el documento no se inyectó al prompt), el LLM tiene casi un 100% de probabilidad de alucinar.
   * *Contras:* Es ciego a la posición. Premia exactamente igual si el motor de búsqueda puso la respuesta en el Rank #1 que si la escondió en el Rank #10.
2. **MAP@K (Mean Average Precision):**
   * *Concepto:* Promedia la precisión acumulada cada vez que se encuentra un documento relevante a lo largo del ranking.
   * *Cálculo:* Es el promedio del *Average Precision* (AP) de todas las consultas. El AP suma la precisión obtenida en cada posición `i` (hasta `K`) donde aparece un documento relevante, y se divide por el total de documentos relevantes esperados.
   * *Pros:* Es muy sensible al orden de llegada. Penaliza matemáticamente a los motores de búsqueda que "entierran" la respuesta correcta al fondo de la lista.
   * *Contras:* Sus puntuaciones no son lineales y es difícil de interpretar intuitivamente para stakeholders de negocio.
3. **NDCG@K (Normalized Discounted Cumulative Gain):**
   * *Concepto:* Es nuestra **métrica directriz**. Asigna una "ganancia" a la relevancia que se degrada o descuenta logarítmicamente conforme el documento cae de posición. El resultado se normaliza entre 0 y 1.
   * *Cálculo:* Suma la relevancia de cada documento recuperado, pero la divide por un penalizador logarítmico basado en su posición (`Relevancia / log2(posición + 1)`), obteniendo el *DCG*. Luego, este valor se divide entre el *Ideal DCG* (el puntaje máximo posible si el orden fuera perfecto) para normalizarlo.
   * *Pros:* Es el estándar de oro en la industria (Google/Bing). Premia masivamente colocar el resultado perfecto en el Rank #1, lo cual es vital en RAG para evitar el efecto de "Lost in the Middle" (donde el LLM ignora texto a la mitad del prompt).

**Métodos de Evaluación del Hit (¿El fragmento contiene la respuesta?):**
1. **A. Subcadena Exacta (`exact_match`):**
   * *Cómo funciona:* Verifica algorítmicamente si una llave (ej. "Anexo 12") existe literalmente dentro del texto del chunk recuperado.
   * *Pros y Contras:* Es ultrarrápida y cuesta $0 USD, ideal para CI/CD continuo. Sin embargo, es engañosa y frágil: si el chunk recuperado dice "Anexo doce", el sistema la califica como error (Falso Negativo).
2. **B. Revisión Manual (Human-in-the-loop):**
   * *Cómo funciona:* Las consultas se exportan a Excel para que un analista experto de la DISF lea el Top 10 y audite la relevancia real.
   * *Pros y Contras:* Es el patrón oro indiscutible. La gran contra es que no escala; evaluar miles de resultados tras tunear un parámetro tomaría semanas de tiempo humano.
3. **C. LLM como Juez (`llm_judge`):**
   * *Cómo funciona:* Se inyecta la Respuesta Ideal (Ground Truth) y el chunk recuperado a un modelo cognitivo (`gpt-4o-mini`), pidiendo un dictamen binario (SÍ/NO) sobre la completitud semántica.
   * *Pros y Contras:* Es altamente escalable, soporta parafraseo y permite simular el "Human-in-the-loop" a nivel de máquina. Su desventaja son los costos de API y el sesgo de calibración (Leniency Bias).

**Hallazgos Evaluativos Avanzados:**
- **Diversidad Cuantificada:** Al fusionar BM25 y Embeddings vía RRF, se demostró matemáticamente baja Correlación de Pearson, comprobando que ambos cometen errores distintos y se complementan.
- **Sesgo de Benevolencia (Leniency Bias):** Se comprobó que `llama3.1` (8B) operando como Juez aprobaba el 37.6% de sus propias alucinaciones, mientras que `gpt-4o` detectaba rigurosamente los fallos, motivando la delegación de auditorías complejas a la nube.
- **Significancia Estadística:** Se implementó *Bootstrapping* (1000 iteraciones) para garantizar que las mejoras de nuestra arquitectura Cascade sobre la línea base no fueran fruto del azar (Intervalo de confianza 95%).

### 3.2 Diagnóstico y Corrección de Pérdida de Contexto (Context Loss)

**El Problema:** Inicialmente, el sistema arrojó un Recall de 27.78%. El diagnóstico reveló "pérdida de contexto jerárquico" (orfandad semántica). Fórmulas profundas quedaban separadas de su cartera original (ej. Crédito Automotriz).
**La Solución:** Inyectar el contexto antes de vectorizar mediante un patrón Pipeline/Filtros en `chunking.py`:
1. **Inyección de Metadatos:** Concatena títulos extraídos físicamente al inicio del chunk (`[Contexto: Cartera Automotriz]`).
2. **Contextual Retrieval (SOTA):** Durante el indexado, un LLM económico lee el documento completo y el chunk, redactando una oración de contexto que se antepone al texto. Esto soluciona la ambigüedad semántica definitivamente. (Implementado con `asyncio` para viabilidad en producción).

### 3.3 Resultados de la Arena de Modelos (La Frontera Eficiente)

Medimos la precisión a través de tres escenarios usando `llm_judge`:
1. **Baseline (Only Chunking):** Recall@10 del **60.0%**. Sufre orfandad semántica.
2. **Inyector de Metadatos:** Recall@10 del **66.67%**. Mejora inmediata al obligar al modelo a "leer" la jerarquía.
3. **Contextual Retrieval (State of the Art):** Recall@10 de **73.33%**. El salto cualitativo masivo que elimina la ambigüedad.
4. **El "Súper RAG" (Maximum Recall):** Búsqueda Híbrida (RRF) expandida simultáneamente con Multi-Query y HyDE, y re-ordenada por Cross-Encoder. Alcanza **90.0%**.

La estrategia base de producción es **Embeddings Puros (con Contextual Retrieval)** por alcanzar 73.33% con latencias de milisegundos. El "Súper RAG" queda a demanda en la interfaz para consultas complejas.

### 3.4 Arquitectura Definitiva: Router Cascade Local-First

**Justificación del Modelo Local y la Frontera de Pareto:** 
En ingeniería de Inteligencia Artificial, la "Frontera de Pareto" es un concepto de optimización multi-objetivo. Imagina un gráfico donde el Eje X representa el *Costo Computacional (VRAM / Latencia)* y el Eje Y representa la *Precisión / Capacidad Cognitiva*. Se dice que un modelo está "en la frontera" si es imposible encontrar otro modelo que sea más inteligente sin ser más costoso, o que sea más barato sin ser más tonto.
*   **Interpretación en ARIF:** Al evaluar motores locales, descartamos modelos colosales (como Llama 70B o Mixtral) porque, aunque ganan márgenes mínimos en precisión, el costo de hardware se dispara exponencialmente (requiriendo clústeres de GPUs caras). Por el contrario, los modelos demasiado pequeños perdían coherencia al redactar español jurídico.
*   **La Decisión (El "Sweet Spot"):** `LLaMA 3.1 (8B)` se sitúa exactamente en el punto de inflexión óptimo de esta frontera. Ofrece una ventana de contexto masiva (128k tokens) y capacidades de razonamiento SOTA (State of the Art) para su tamaño, exigiendo escasos 4.5 a 8GB de VRAM. Esto permite operarlo fluidamente en entornos locales (Ollama) sin fricción. Además, en las pruebas superó holgadamente a Mistral 7B en la asimilación del tono normativo oficial.

Para la privacidad institucional de Banxico, implementamos la política **Local-First** mediante un **Model Cascading Heterogéneo**:
1. **Intento Local (Llama 3.1):** El sistema interroga al modelo local gratuito.
2. **Autoevaluación (Faithfulness):** El sistema evalúa la fidelidad de su propia respuesta extrayendo claims contra el contexto.
3. **Escalado Dinámico:** Si la confianza es alta (>=0.80), la entrega. Si es baja, la desecha y redirige silenciosamente a la nube (`gpt-4o-mini`).
Esto reduce el TCO en un >80% derivando el "Happy Path" a modelos gratuitos sin penalizar calidad.

**Ollama vs vLLM:** En desarrollo local, se usa Ollama mediante Patrón Factory (`config_llm.py`). Para producción, la arquitectura contempla migrar a vLLM en Docker (AWS/GCP con GPUs L4) para soportar procesamiento por lotes continuo bajo alta concurrencia.

### 3.5 Anatomía del Pipeline de Recuperación (Retrieval Pipeline)

El proceso de responder a una consulta no es un flujo lineal simple. Para maximizar la precisión, la implementación central (`src/nlp_core/pipeline.py`) opera como un orquestador modular dividido en tres fases altamente desacopladas:

1. **Fase 1: Pre-procesamiento y Expansión (Query Expansion)**
   *   *¿Qué hace?* Intercepta la pregunta original del usuario y la altera dinámicamente antes de tocar la base de datos para superar las barreras del lenguaje humano.
   *   *¿Cómo funciona?* Permite activar *Multi-Query* (generación de paráfrasis) o *HyDE* (generación de un documento respuesta alucinado). Si el usuario lo requiere, ambas técnicas pueden dispararse en conjunto. El resultado es que una sola pregunta humana se convierte en múltiples "hilos" de búsqueda matemáticos.
2. **Fase 2: Recuperación Base (Base Retrieval)**
   *   *¿Qué hace?* Desciende a los motores de búsqueda para extraer los fragmentos crudos (chunks).
   *   *¿Cómo funciona?* Recibe la batería de consultas generadas en la Fase 1. El motor de búsqueda es intercambiable en caliente (puede ser TF-IDF, BM25 en RAM, Embeddings en ChromaDB, o un Híbrido RRF). El pipeline ejecuta búsquedas exhaustivas por cada hilo generado y luego los **consolida y desduplica** utilizando algoritmos de hashing en memoria (evitando que el mismo artículo se repita). Si la Fase 3 está activa, el motor extraerá intencionalmente una red masiva de candidatos (ej. Top 20) en lugar del Top 5 usual, para garantizar que haya suficiente materia prima.
3. **Fase 3: Post-procesamiento y Reordenamiento (Reranking)**
   *   *¿Qué hace?* Filtra la red masiva de documentos candidatos, descartando el ruido y ordenando los verdaderos diamantes para entregárselos al LLM Generador final.
   *   *¿Cómo funciona?* Utiliza un modelo neuronal especializado conocido como **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`). A diferencia de la Búsqueda Vectorial bi-encoder (que calcula distancia entre dos puntos pre-vectorizados), el Cross-Encoder lee el par exacto *(Pregunta Original + Documento Candidato)* de manera simultánea, permitiéndole entender la intención profunda. Emite una puntuación de relevancia ultra-precisa, reordena la lista de mayor a menor, y finalmente recorta únicamente el `Top K` estricto (ej. Top 5) solicitado por el usuario.


## 4. Módulo de Extracción Estructurada y Gobernanza de Datos

Retomando la meta de diseñar formularios estandarizados y auditar el repositorio masivo, el sistema implementa flujos de extracción estructurada independientes al chat principal.

### 4.1 Data Provenance y el Estándar de Oro (`manifest.yaml`)
En un RAG institucional, si el LLM responde basándose en un fragmento de texto, necesitamos trazabilidad absoluta (¿De qué documento salió? ¿De qué institución? ¿Está vigente?). Sin esto, la base vectorial degeneraría en un "Data Swamp" (pantano de datos) inauditable.
*   **El Manifiesto:** Implementamos un archivo `manifest.yaml` como la única fuente de la verdad (*Single Source of Truth*) para inyectar los metadatos en ChromaDB. Al ser texto plano, permite versionamiento perfecto en Git, habilita pre-filtros de seguridad de acceso por rol (RBAC), y garantiza que el motor jamás recupere o mezcle normas derogadas.

### 4.2 Casos de Uso Core: Autogeneración y Formularios
Llenar un manifiesto manualmente o deducir los campos requeridos en una base de datos exige cientos de horas humanas. Para automatizarlo, el sistema despliega dos flujos (disponibles en la App):
1. **Autogeneración de Metadatos de Gobernanza:** Un analista sube un PDF regulatorio. El LLM asume el rol de "Bibliotecario" y deduce automáticamente la Ficha Técnica (Tema, Institución, Nivel de Confidencialidad) emitiendo el JSON listo para inyectarse al `manifest.yaml`.
2. **Generación de Formularios desde Normativa:** En lugar de leer toda la ley para saber qué solicitarle a los bancos, el LLM actúa como "Analista de Datos", infiriendo el esquema relacional requerido (Nombres de campo, Catálogos, Tipos de dato) a partir de un extracto subido.

### 4.3 Paradigmas: RAG vs Long-Context y Structured Outputs
Los LLMs tienden a responder con texto libre hiper-verboso. Para la extracción descrita arriba usamos **Structured Outputs** nativos de OpenAI (`client.beta.chat.completions.parse`), esposando al modelo a moldes Pydantic (`MetadataDocumento` o `FormularioEstructurado`) que devuelven JSON estrictos y predecibles.
*   **Por qué NO usamos RAG:** El RAG es un "buscador quirúrgico" ideal para responder preguntas. Pero para extraer la estructura de un formulario o el tema de un documento, el RAG fallaría al recuperar solo trozos asilados, omitiendo datos esparcidos en páginas distintas.
*   **La Solución (Long-Context Extraction):** Para la extracción, esquivamos la búsqueda vectorial por completo y pasamos el documento íntegro a la ventana masiva del modelo (hasta 128k tokens). Esto le da una visión panorámica al LLM, permitiéndole entender referencias cruzadas y extraer la estructura de manera holística y perfecta.
*   **Extracción Guiada (Prompt Injection Controlado):** Recientemente introdujimos la capacidad de que el usuario inyecte instrucciones específicas (ej. "extrae exclusivamente temas de riesgo crediticio") directamente en el prompt del usuario. Esto guía al modelo sin alterar el "System Prompt" maestro ni romper el esquema Pydantic. **Consideración de Seguridad Futura:** Al abrir una ventana de texto libre directo al prompt, el sistema se vuelve susceptible a ataques de *Prompt Injection* maliciosos. Para que esta característica sea totalmente segura y funcional en un entorno productivo, será necesario implementar una capa de validación de entradas (Input Guardrails) estricta antes de la inferencia.

### 4.4 Escalabilidad hacia Enterprise (Map-Reduce)
Para mitigar que el enfoque *Long-Context* sufra el síndrome de "Lost in the Middle" (cuando los LLMs olvidan texto atrapado a la mitad de un mega-documento) frente a leyes colosales de cientos de páginas, se diseñó la arquitectura Map-Reduce:
1. **Map:** Pre-procesar el documento dividiéndolo por sus Capítulos naturales y lanzar múltiples extracciones en paralelo.
2. **Reduce:** Un LLM consolidador final recibe los mini-esquemas de todos los hilos, elimina duplicados y ensambla un esquema maestro inquebrantable.

## 5. Demostración Técnica: Interfaces de Usuario y Operaciones (MLOps)

Para disponibilizar toda la robustez de la arquitectura, se construyeron dos clientes:

### 5.1 Aplicación de Usuario Final (Analistas DISF)
Ubicada en `app/` (Vanilla JS, HTML, CSS), permite tres modalidades:
1. **Chat Normativo:** RAG con un HUD que expone latencia, ruta Cascade, y ruteo documental. Soporta matemáticas con `KaTeX`.
2. **Extracción de Formularios:** Tablas HTML dinámicas que mapean el output de Pydantic.
3. **Extracción Dinámica de Metadatos:** Clasifica confidencialidad y audiencia.

**Búsqueda Dinámica Efímera:** El analista puede arrastrar un PDF temporal (oficio, acta) para procesarlo en memoria RAM, habilitando un *EnsembleRetriever* que cruza el documento del usuario contra la base oficial histórica sin contaminarla.

### 5.2 Centro de Comando MLOps (Ingenieros)
Ubicado en `dashboard/app_evaluaciones.py` (Streamlit), es el cuarto de control operativo:
1. **Telemetría y FinOps:** Rastreador de latencia de cola larga (P50/P90/P99) y tokens, persistidos en `telemetria_llm.jsonl`.
2. **Taxonomía de Errores y ECE:** Ejecución de pruebas RAGAS bajo demanda y Calibración Inyectando alta temperatura para detectar alucinaciones encubiertas.

### 5.3 Auditoría de Caja Blanca (Trazabilidad RAG)
El concepto de "caja negra" es inaceptable regulatoriamente. Un `semantic_cache.pkl` captura la anatomía de cada inferencia:
- **Trazabilidad:** Permite inspeccionar qué prompt exacto se usó y qué fragmentos se inyectaron en el pasado.
- **Caché Semántico con Juez LLM:** Para optimizar respuestas repetidas ante variaciones gramaticales, un Juez LLM local evalúa la equivalencia de intención (SÍ/NO) de la consulta actual contra el caché, devolviendo respuestas en ~0.5s y evadiendo la latencia completa de RAG.

## 6. Seguridad, Recomendaciones y Horizonte de Escalabilidad (El Futuro)

### 6.1 Seguridad Proactiva y Jerarquía de la Verdad
- **Guardrails (Input):** Un Juez de Entrada detecta *Prompt Injection* y *Jailbreaks*, bloqueando peticiones maliciosas en el perímetro sin gastar tokens.
- **Jerarquía Jurídica:** Ante conflictos (Ley vs Guía), se aplicó *Prompt Engineering Jerárquico*: "La normativa oficial tiene prelación absoluta", permitiendo resolución lógica sin alterar los scores matemáticos del motor de búsqueda.

### 6.2 Arquitectura Cloud y FinOps (Avance 6)
- **TCO (DEP-B):** Análisis del costo total comparando la solución Self-hosted frente al esquema Multi-Cloud.
- **Monitoreo en Producción:** Establecimiento de SLOs (ej. P95 <= 3.5s) y alertas de *Drift Detection*.
- **Plan de Handoff (DEP-E):** Entrega documentada de artefactos serializados (ChromaDB, Prompts JSON) para un *Decommissioning plan* seguro institucional.

### 6.3 Horizonte Tecnológico (Auto-optimización)
- **Paradigma DSPy:** Se validó una prueba de concepto para superar el *Prompt Engineering* manual. Un optimizador empaqueta iterativamente los mejores ejemplos matemáticamente en el prompt, evolucionando a un sistema de auto-mejora continua.
- **Control de Accesos (RBAC):** Uso de metadatos en ChromaDB para restringir inyección de contexto según el perfil del analista.

### 6.4 Resumen de Beneficios y Roadmap Final
El proyecto ARIF reduce tiempos de consulta drásticamente, democratiza el conocimiento experto de la DISF y minimiza el riesgo de errores de interpretación.

El seguimiento detallado a futuro se maneja en el documento **[`support/pendientes_y_roadmap_final.txt`](../../support/pendientes_y_roadmap_final.txt)**, e incluye:
1. Estructurar un piloto institucional a mayor escala (Hardware profiling).
2. Robustecer la gobernanza de datos.
3. Reactivar el diseño automatizado de formularios incorporando estrategias Human-in-the-Loop (HITL) para validaciones expertas sobre inferencias de IA.

## 7. Visión Global de la Arquitectura y Desglose de Scripts

Para tener una perspectiva panorámica completa, la solución ARIF no es un script monolítico, sino un ecosistema modular de microservicios. A grandes rasgos, la implementación física (`src/`, `app/`, `dashboard/`) opera en cinco grandes macrosistemas:

### 7.1 El Módulo de Ingesta (El "ETL" Vectorial)
Se encarga de transformar el caos de los documentos regulatorios en datos matemáticos estructurados y limpios.
*   **`src/ingesta/ingestor.py`**: El motor de extracción. Extrae texto de PDFs (vía OCR) o Excels sin importar el formato original.
*   **`src/utils/limpieza_texto.py`**: Elimina el "ruido" institucional (ej. firmas del DOF, sellos, pies de página irrelevantes).
*   **`src/nlp_core/chunking.py`**: Fragmenta los textos respetando la jerarquía legal e inyecta memoria asíncrona mediante *Contextual Retrieval*.
*   **`src/nlp_core/vectorizacion.py`**: Interfaz con ChromaDB. Transforma los fragmentos de texto en embeddings matemáticos y los persiste en la base de datos.

### 7.2 El Motor Central NLP (Cerebro de Búsqueda y Generación)
El núcleo que orquesta el razonamiento de la inteligencia artificial.
*   **`src/nlp_core/seguridad/guardrails.py`**: Actúa como un cortafuegos. Intercepta ataques, *Jailbreaks* o toxicidad financiera antes de procesar la consulta.
*   **`src/nlp_core/generacion.py`**: El orquestador principal. Implementa el *Router Cascade* (Llama 3.1 vs GPT-4o-mini) y el Caché Semántico.
*   **`src/nlp_core/retrieval.py` y `pipeline.py`**: Ejecutan el motor híbrido (BM25 + ChromaDB con RRF), despliegan las estrategias de expansión de la consulta (Multi-Query/HyDE), y filtran la respuesta óptima mediante un *Cross-Encoder* profundo.
*   **`src/nlp_core/config_llm.py` y `prompts_registry.py`**: Manejan la inyección de los modelos (Ollama/OpenAI) y los "System Prompts" firmados criptográficamente para trazabilidad legal.

### 7.3 El Módulo de Extracción Estructurada
Un flujo alterno para automatización rígida.
*   **`src/nlp_core/schemas.py`**: Define los moldes Pydantic estrictos (ej. `RequerimientoInformacion`).
*   **`generacion.py (Vía Structured Outputs)`**: En lugar de platicar libremente, este flujo obliga al LLM a leer documentos inmensos y devolver exclusivamente un JSON estructurado, inquebrantable y sin alucinaciones, permitiendo la automatización de procesos.

### 7.4 El Laboratorio de MLOps (El Sistema Evaluador)
Una suite para pruebas automatizadas, optimización y justificación del retorno de inversión (ROI).
*   **`src/nlp_core/evals/evaluador.py`**: Actuando como un científico en las sombras, bombardea al sistema con consultas humanas reales y usa a un LLM avanzado como Juez (`llm_judge`) para puntuar automáticamente si el bot local se equivocó o alucinó.
*   **`src/nlp_core/telemetria.py`**: Registra en un log silencioso los tiempos de latencia y los centavos de dólar gastados por cada petición, información crítica para el control financiero.
*   **`src/lab/generar_pareto_final.py` y `calcular_delta_ma4.py`**: Scripts de ciencia de datos que cruzan los resultados y ejecutan simulaciones matemáticas (Bootstrapping) para dibujar gráficas que demuestran qué modelo es el más eficiente (Frontera de Pareto).

### 7.5 Interfaces de Usuario (Frontend, Backend y Analítica)
Donde el usuario final y los ingenieros interactúan con la "caja negra" de la IA.
*   **`api/main_api.py`**: Expone los endpoints robustos (FastAPI) para consumir el NLP Core de forma programática.
*   **`app/index.html` y `app/main.js`**: La interfaz web interactiva del usuario final (la "App"). Permite entablar el chat normativo, subir documentos efímeros para analizarlos al vuelo, y visualizar las citas de origen exactas.
*   **`dashboard/app_evaluaciones.py`**: Un panel de control construido en Streamlit exclusivo para el equipo de MLOps. Lee la telemetría, mapea la taxonomía de errores del evaluador, y visualiza de forma interactiva las gráficas de rendimiento sin tener que tocar la base de código.

### 7.6 Gestión de Datos, Prompts y Artefactos
La inteligencia del sistema no vive solo en el código, sino en los archivos de configuración, memoria y evaluación.
*   **`src/nlp_core/prompts.json`**: El corazón del comportamiento. En lugar de tener las instrucciones del LLM dispersas por el código (Hardcoded), este JSON centraliza todas las "personalidades" (ej. `qa_system`, `contextual_chunking`, `hyde`, `guardrails_toxicity`). Al combinarse con `prompts_registry.py`, permite versionar el comportamiento del modelo de forma independiente al código, y asegura criptográficamente que no se alteren en producción sin autorización.
*   **`data/evaluacion_dataset.json`**: El *Ground Truth*. Contiene el banco de más de 100 consultas humanas curadas y respondidas por expertos de la DISF, sirviendo como la "hoja de respuestas correctas" para el evaluador automático.
*   **`data/config_experimentos.json`**: Orquesta el laboratorio. Define qué variaciones del RAG se van a probar (ej. Baseline Léxico vs Híbrido Simple vs SOTA_Completo).
*   **`data/03_output/`**: El repositorio de artefactos vivos generados por el sistema:
    *   `telemetria_llm.jsonl`: Archivo transaccional inmutable (append-only) que anota silenciosamente cada *token* gastado y cada latencia de inferencia en milisegundos.
    *   `semantic_cache.pkl`: El caché vectorizado que almacena las preguntas frecuentes para responder en menos de 0.5s en la Interfaz de Usuario sin ir al LLM Generativo.
    *   `evaluaciones_*/` y `graficos/`: Carpetas dinámicas donde `evaluador.py` deposita las calificaciones brutas, y donde `generar_pareto_final.py` dibuja las gráficas `.png` que terminan justificando las decisiones arquitectónicas.
*   **`src/utils/` y `src/lab/`**: Scripts de soporte vitales. Aquí viven herramientas de mantenimiento como `poblar_chroma.py` (para ingestar la base documental completa offline), `exportar_auditoria_manual.py` (para sacar excels legibles por humanos ante errores del LLM), y `analisis_texto.py` para utilería.
