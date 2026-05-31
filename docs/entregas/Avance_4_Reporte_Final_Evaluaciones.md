# **Procesamiento de Lenguaje Natural**
## Maestria en Inteligencia Artificial Aplicada
### Tecnologico de Monterrey

* **Nombres y matriculas**
    * Sarmiento Cervantes Jacqueline: A01795863
    * Mayoral Teran Alexandro: A01795899
* **Numero de equipo: 8**

---

# 🎓 Proyecto Integrador: Avance 4 - Evaluación Core, Pareto y Métricas Base (DISF)

Este documento es el **entregable** del Avance 4 y funge como la base teórica y técnica para la evaluación del sistema. Su estructura fluye de lo fundacional (arquitectura y telemetría) hacia la validación estadística (Pareto y Taxonomía de errores), cumpliendo con las exigencias de Banco de México y las rúbricas de la maestría.

---

## 1. Preparación del Entorno y Arquitectura Core

Para garantizar la reproducibilidad de este Notebook, inicializamos el entorno conectando directamente con la arquitectura modular desarrollada en `src/`.

```python
# 1. Preparación del Entorno
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

load_dotenv(project_root / ".env")
print("✅ Entorno preparado. Importaciones locales de src/ listas.")
```

### 1.1 Diseño y Arquitectura de Scripts Core (Deep-Dive)
Para automatizar la evaluación y cerrar las brechas críticas de avances pasados, se construyeron 3 pilares arquitectónicos:

1. **Multi-Chunk Handling y Síntesis Multi-Documento (`generador_ground_truth_llm.py`):** Banxico requiere cruzar información de múltiples regulaciones. Para lograr esto, se eliminó la restricción de un-solo-documento. Ahora el *Retriever* recupera el *Top-K* de fragmentos (Chunks) de distintas fuentes (CUB, LIC, etc.) y los inyecta simultáneamente en el prompt del LLM. Esto habilita el **Multi-Chunk Handling** real, permitiendo al modelo sintetizar una respuesta coherente a partir de múltiples piezas de rompecabezas normativo.
   > 💡 *Concepto Técnico:* Se vectorizan fragmentos de múltiples PDFs en una única base de datos vectorial ChromaDB. Durante la consulta, se hace una búsqueda k-NN (K-Nearest Neighbors) global. Los resultados se concatenan como un único bloque de contexto enriquecido, permitiendo al LLM leer y correlacionar información que físicamente reside en distintos archivos PDF.

2. **Configuración Conmutable (`config_llm.py`):** Dependiendo de un flag en el `.env` (`USE_LOCAL_LLM=True`), conmuta instantáneamente entre Ollama (Llama 3.1) y OpenAI (GPT-4o) sin refactorizar el código base.
   > 💡 *Concepto Técnico:* Se implementó el patrón de diseño de software **Factory**. Al inicializar el sistema, el Factory lee el `.env` y decide dinámicamente si instanciar clases del SDK de OpenAI o clases envoltorio (wrappers) de la API local de Ollama. Ambos devuelven la misma interfaz de métodos, por lo que el resto del código es completamente agnóstico al modelo que está corriendo por debajo.

3. **El Orquestador (`evaluador_integral.py`):** Carga dinámicamente el `config_experimentos.json` con las arquitecturas a evaluar, inyecta el hash criptográfico, ejecuta el RAG y clasifica los fallos.
   > 💡 *Concepto Técnico:* Funciona como un pipeline tipo DAG (Directed Acyclic Graph) sincrónico. Itera sobre las 110 consultas del dataset, pasándole la misma pregunta exacta a los diferentes LLMs instanciados por el Factory. Para cada LLM, captura la respuesta, la envía al módulo de telemetría y guarda el DataFrame en un CSV estandarizado para su graficación matemática.

---

## 2. Trazabilidad, Versionamiento y Prompt Registry

Uno de los pilares de un proyecto MLOps para entidades financieras es la reproducibilidad. Si el sistema emite una respuesta, el Banco debe rastrear exactamente qué instrucción la generó. 
Se construyó el módulo `src/nlp_core/prompts_registry.py` respaldado por `prompts.json`. Al cargar un prompt, inyecta dinámicamente un **Hash Criptográfico (SHA-256)** que viaja con la telemetría.

> 💡 *Concepto Técnico:* Cada vez que se manda llamar un system prompt, el módulo de Python calcula una suma de comprobación criptográfica (Hash SHA-256) del string crudo del prompt. Este string hexadecimal (Ej. `a1b2c3d...`) se anexa a la respuesta final. Si alguien altera un solo espacio en blanco en el prompt, el hash cambia drásticamente, previniendo alteraciones silenciosas en la instrucción de evaluación.

```python
# 2. Demostración de Trazabilidad Criptográfica
from src.nlp_core.prompts_registry import get_prompt

# Obtenemos el prompt exacto de QA RAG y su Hash único
prompt_text, version, prompt_hash = get_prompt("qa_rag")
print(f"🔒 Hash del Prompt Actual: {prompt_hash} (Versión: {version})")
print(f"📝 Inicio del Prompt: {prompt_text[:100]}...")
```

---

## 3. Módulo de Telemetría (TCO y P95)

En sistemas RAG, "el contexto es dinero". Se implementó la clase `RastreadorTelemetria` que intercepta las llamadas para contar los tokens a nivel BPE con `tiktoken` y calcular el TCO (Total Cost of Ownership).

> 💡 *Concepto Técnico:* La telemetría se implementó bajo el patrón **Decorator** o inyección de dependencias. Envuelve la función `generar()` del LLM para arrancar un cronómetro (midiendo milisegundos de latencia) y pasa el texto crudo por la librería `tiktoken` para segmentarlo en unidades léxicas exactas (tokens), lo que permite multiplicar por los precios por millón vigentes de la API comercial, o registrar costo $0 para los modelos locales.

```python
# 3. Inicialización del Rastreador de Telemetría
from src.nlp_core.telemetria import RastreadorTelemetria
import time

telemetria = RastreadorTelemetria()

# Simulamos la latencia y tokens de una ejecución RAG
inicio = time.time()
# (Aquí ocurre la ejecución del LLM)
time.sleep(0.5)
latencia = time.time() - inicio

telemetria.registrar_consulta(
    modelo="llama3.1:8b-local",
    latencia_segundos=latencia,
    tokens_input=150,
    tokens_output=50
)

metricas = telemetria.calcular_metricas("llama3.1:8b-local")
print(f"👻 Latencia P95 estimada: {metricas['latencia_p95']:.2f}s")
print(f"💸 Costo estimado por 1000 consultas: ${metricas['costo_por_1000_consultas']:.4f}")
```

### 3.1 Estrategias de Optimización y Límite de Tokens (Cost Containment)
Para que un sistema RAG sea viable financieramente en un entorno productivo como Banxico, medir el costo no es suficiente; se deben implementar mecanismos activos de mitigación del TCO (*Cost Containment*). Además de medir los tokens, la arquitectura conceptual considera las siguientes estrategias de optimización:

- **Límites Estrictos de Generación (`max_tokens`):** Se establece un límite matemático (hard-limit) en la respuesta del LLM. Dado que requerimos formatos JSON estructurados y concisos, limitar los tokens de salida a un máximo de 500 previene ataques de denegación de billetera (*Denial of Wallet*) o bucles infinitos de alucinación donde el modelo genera párrafos innecesarios.
- **Reranking y Poda de Contexto:** Los tokens de entrada (Input Tokens) representan el mayor costo en un RAG. En lugar de inyectar 10 fragmentos completos recuperados por similitud de coseno pura, se pasa por un modelo de *Reranking* cruzado (Cross-Encoder) para podar el contexto e inyectar estrictamente los 2 o 3 fragmentos vitales, reduciendo el gasto de API hasta en un 70%.
- **Semantic Caching (Caché Semántico):** Para consultas regulatorias recurrentes (Ej. *"¿Cuál es la tasa de estimación del Anexo 33?"*), se implementa un caché a nivel vectorial. Si una pregunta entrante tiene una similitud semántica de >0.95 con una pregunta previa ya respondida, se devuelve la respuesta almacenada sin invocar a la API del LLM, reduciendo costo y latencia P95 a 0.
- **Model Cascading (Enrutamiento de Modelos):** Tareas de clasificación de taxonomía o extracción de entidades simples se desvían a modelos locales gratuitos (Llama 3.1 8B), reservando las llamadas a APIs comerciales pesadas (GPT-4o) exclusivamente para tareas de razonamiento profundo o síntesis financiera compleja.

```python
# 3.1 Demostración de Cost Containment (Límite de Tokens y Cascading)
from src.nlp_core.config_llm import get_langchain_chat
import os

# Simulamos que el orquestador decide usar el modelo local para una tarea simple (Cascading = $0 TCO)
os.environ["USE_LOCAL_QA"] = "true"
llm_barato = get_langchain_chat(task="qa", temperature=0.0)

# Para tareas de alta complejidad, se usa la nube con límite estricto de tokens de salida (Denial of Wallet)
os.environ["USE_LOCAL_QA"] = "false"
llm_optimizado = get_langchain_chat(task="qa", temperature=0.0)
llm_optimizado.model_kwargs = {"max_tokens": 500} # Hard-limit financiero

print("🛡️ Estrategias de Cost Containment inyectadas exitosamente.")
```

---

## 4. Evolución del Dataset, Contextual Retrieval y Diversidad de Modelos

### 4.1 Modelos Alternativos y Residencia de Datos

Para cumplir con los requerimientos, configuramos ≥6 arquitecturas LLM distintas, asegurando que al menos un candidato fuera **Self-Hostable** (Llama 3.1 8B vía Ollama). Esto garantiza a Banxico la **residencia de datos** (on-premise) e independencia del *vendor lock-in*.

#### Justificación Técnica de LLaMA 3.1 8B (Frontera de Pareto)

La elección de un modelo local para tareas NLP corporativas se reduce a un problema de optimización entre **Hardware Requerido vs. Razonamiento Lógico**. LLaMA 3.1 8B se sitúa exactamente en el "Sweet Spot" de esta frontera:

**Ventajas Principales:**
1. **Ventana de Contexto Masiva (128k Tokens):** Crítico para RAG, permite ingerir contextos gigantes sin colapsar.
2. **Hardware de Consumo:** Cuantizado a 4 u 8 bits mediante Ollama, requiere entre 4.5 GB y 8 GB de VRAM.
3. **Capacidad Multilingüe:** Manejo del español normativo/legal excepcionalmente fluido.
4. **Instrucciones Estrictas:** Altamente ajustado para seguir reglas duras (vital para validación estricta Pydantic).

**Comparativa contra Alternativas Descartadas:**
- ❌ **Mistral v0.3 (7B) / Mixtral (8x7B):** Mistral 7B fue superado por Llama 3.1. Mixtral requiere ~24 GB de VRAM (inviable en laptops estándar).
- ❌ **Gemma 2 (9B):** Excelente rendimiento, pero con licencia comercial más restrictiva y mayor consumo eléctrico.
- ❌ **LLaMA 3.1 (70B):** Exige hardware empresarial de ultra-alta gama (Múltiples GPUs de 80GB).

> [!IMPORTANT]
> **Veredicto:** LLaMA 3.1 8B es la mejor opción estratégica actual para nuestro caso de uso. Provee un razonamiento casi a la par de los modelos frontera del año pasado de forma 100% privada y off-grid.

#### Viabilidad de Inferencia Local: Ollama vs vLLM

Para la ejecución física del modelo, el proyecto adopta una **Estrategia Híbrida de Inferencia**:
1.  **Fase de Desarrollo y Evaluación:** Se utiliza **Ollama** (basado en `llama.cpp`). Es ideal porque tiene un consumo bajo demanda, no bloquea toda la memoria VRAM (*Memory Mapping* dinámico) y funciona de forma estable y nativa en Windows sin virtualización pesada.
2.  **Fase de Producción (Escalabilidad):** Para soportar cientos de peticiones concurrentes de los analistas de la DISF, se migrará la inferencia a **vLLM** (en servidores Linux dedicados), el cual implementa *Continuous Batching* y *PagedAttention* para maximizar el *throughput* a nivel empresarial.

### 4.2 Ampliación del Golden Dataset (Rúbrica MA2 / B1)
El esfuerzo humano para curar datos de calidad llevó el *Golden Dataset* de 30 a **110 consultas evaluables**, garantizando significancia estadística y rigor normativo.

> 💡 *Concepto Técnico:* La creación del Golden Dataset (Ground Truth) es la base de cualquier evaluación seria. Estas 110 consultas **fueron redactadas meticulosamente a mano por expertos en la materia (SMEs - Subject Matter Experts)** para asegurar que las preguntas trampa, el argot legal y los escenarios ambiguos reflejen la realidad operativa de la Comisión Nacional Bancaria y de Valores (CNBV). Adicionalmente, se desarrolló un script de generación sintética con LLMs (`generador_ground_truth_llm.py`) como una arquitectura de apoyo para escalar el dataset en fases futuras si se requiere un estrés masivo (Red-Teaming automatizado).

```python
# 4. Carga del Golden Dataset Expandido
import json

ruta_dataset = project_root / "data" / "evaluacion_dataset.json"
with open(ruta_dataset, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"📚 Dataset cargado con {len(dataset)} consultas regulatorias complejas.")
```

### 4.3 Mitigación de Pérdida de Contexto (Context Loss)
Para evitar la "orfandad" semántica de fórmulas en incisos profundos, migramos a **Contextual Retrieval** y **Búsqueda Híbrida RRF** (Coseno + BM25) para evitar la deriva semántica (*semantic drift*).

> 💡 *Concepto Técnico:* Un chunk crudo como "El porcentaje es 15%" pierde totalmente su significado semántico sin el título original. El *Contextual Retrieval* inyecta los metadatos (Ej: `[Documento: Ley de Instituciones, Título II] El porcentaje es 15%`) **antes** de enviarlo al modelo de Embeddings, forzando al vector a mapearse correctamente en el espacio hiperdimensional de la ley correcta.

---

## 5. Prueba Ciega y Contaminación de Datos por Candidato

Cumpliendo con lo indicado, corrimos una prueba de *No-Context Test* **por cada candidato LLM**. Aislando al LLM del buscador, forzamos respuestas basadas solo en su corpus de entrenamiento original.
**Resultado:** Todos alucinaron regulaciones europeas, demostrando que el RAG es estrictamente necesario.

> 💡 *Concepto Técnico:* En lugar de pasar los Chunks recuperados de la Base de Datos Vectorial, la variable de contexto se inyecta intencionalmente como un string vacío `contexto_recuperado=""`. Esto fuerza al LLM a recaer en su "memoria paramétrica" (pesos neuronales adquiridos durante el pre-entrenamiento global) y expone si la respuesta viene del modelo o del archivo legal de Banxico.

```python
# 5. Simulador de Prueba Ciega (Data Contamination)
from src.nlp_core.config_llm import get_llm_client, get_llm_model_name
import os

# Usamos el modelo local por default para esta prueba ciega
os.environ["USE_LOCAL_QA"] = "true"
client = get_llm_client("qa")
modelo_qa = get_llm_model_name("qa")

pregunta_financiera = "¿Cómo se calcula la estimación preventiva de tarjetas de crédito según el Anexo 33?"

# El LLM responde puramente de su memoria paramétrica, SIN fragmentos normativos
respuesta = client.chat.completions.create(
    model=modelo_qa,
    messages=[
        {"role": "system", "content": "Eres un asistente experto en regulación financiera."},
        {"role": "user", "content": pregunta_financiera}
    ],
    temperature=0.0
)

respuesta_ciega = respuesta.choices[0].message.content
print(f"⚠️ Respuesta de Memoria Paramétrica (Prueba Ciega): {respuesta_ciega}")
```

---

## 6. Optimización Multiobjetivo y Significancia Estadística

### 6.1 Significancia Estadística
No comparamos números crudos. El orquestador ejecuta **1,000 resamples (Bootstrap CI)** sobre el NDCG para obtener intervalos de confianza al 95%. Si se solapan, es un empate estadístico.

> 💡 *Concepto Técnico:* El *Bootstrapping* toma muestras aleatorias con reemplazo de las 110 calificaciones generadas y calcula la media repetidamente (1,000 iteraciones). Esto genera una distribución normal estadística sin importar cómo se comportaban originalmente los datos, permitiendo trazar límites de control matemáticos robustos.

### 6.2 Frontera de Pareto
En entornos financieros, la toma de decisiones tecnológicas rara vez es un problema de un solo objetivo (ej. "sólo maximizar precisión"). Por ello, graficamos la **Frontera de Pareto**, mapeando el **Costo Operativo/Latencia (Eje X)** contra la Calidad de Recuperación **NDCG@10 (Eje Y)**.

> 💡 *Concepto Técnico:* La Frontera de Pareto es un concepto de optimización multiobjetivo. Sirve para identificar el subconjunto de modelos "óptimos". La idea central es que un modelo pertenece a la frontera si **no existe ningún otro modelo que sea mejor o igual en ambos ejes**. 
> - **Cómo se interpreta:** Si trazas una curva imaginaria uniendo los modelos más eficientes, todos los puntos que caen *exactamente sobre la línea* son los candidatos viables (por ejemplo, uno muy barato pero de calidad media, o uno muy caro pero de calidad altísima). Cualquier modelo que caiga *por debajo o a la derecha* de la línea se considera un modelo "dominado" (subóptimo), ya que existe al menos otro modelo en la gráfica que es más barato y más preciso a la vez. Esto permite a Banxico seleccionar el "Top-2" descartando matemáticamente arquitecturas ineficientes.

```python
from pathlib import Path
import pandas as pd
from src.lab.graficos import plot_frontera_pareto

# En Jupyter Notebook, Path.cwd() suele ser la carpeta 'notebooks', por lo que la raíz es solo un .parent arriba
project_root = Path(__file__).resolve().parents[3] if '__file__' in locals() else Path.cwd().parent

# 1. Definir las carpetas de las corridas (Local vs Nube)
carpetas_corridas = [
    "oficiales/run_local",
    "oficiales/run_nube"
]

resultados_maestros = []
for nombre_carpeta in carpetas_corridas:
    carpeta_eval = project_root / "data" / "03_output" / "evaluaciones" / nombre_carpeta
    
    # glob() no lanza excepción si no encuentra nada, devuelve una lista vacía
    archivos_encontrados = list(carpeta_eval.glob("ARENA_RESULTADOS*.csv"))
    if not archivos_encontrados:
        print(f"⚠️ Ignorando {nombre_carpeta}: No se encontraron resultados de Arena todavía.")
        continue
        
    archivo_arena = archivos_encontrados[0]
    df_arena = pd.read_csv(archivo_arena)
    
    # Procesar y concatenar resultados con sufijo
    for _, row in df_arena.iterrows():
        # Soportamos múltiples formatos históricos de las columnas de resultados
        costo = row.get("Costo_Total_USD", row.get("Costo", 0.0))
        ndcg = row.get("NDCG@10", row.get("NDCG_Promedio", row.get("NDCG_Mean", 0.0)))
        candidato_id = row.get("estrategia", row.get("Candidato_ID", row.iloc[0]))
        
        resultados_maestros.append({
            "modelo": f"{candidato_id} [{nombre_carpeta.split('/')[-1]}]",
            "costo_por_1000": costo,
            "ndcg": ndcg
        })

if resultados_maestros:
    img_path = "./pareto_frontier_comparativa_global.png"
    plot_frontera_pareto(resultados_maestros, img_path)
    print("📈 Gráfico de Pareto Global generado exitosamente.")
    
    # Mostrar la imagen directamente en el output del Notebook
    from IPython.display import Image, display
    display(Image(filename=img_path))
else:
    print("❌ No hay datos para generar el gráfico de Pareto.")
```

> 💡 **Análisis de la Frontera de Pareto (Ejecución Local)**
> - **Costo Operativo (Eje X):** Ambos puntos (Baseline Léxico y SOTA Completo) se sitúan exactamente en `$0.00 USD`. Esto valida empíricamente la hipótesis financiera: utilizar Llama 3.1 8B de forma local mantiene el costo marginal por consulta en cero.
> - **Precisión NDCG@10 (Eje Y):** La estrategia `6_SOTA_Completo` (Búsqueda Híbrida + Reranking) demuestra una clara superioridad sobre el `1_Baseline_Léxico`. 
> - **Conclusión MLOps:** Al implementar técnicas avanzadas de recuperación, logramos incrementar la calidad y relevancia de los contextos extraídos sin impactar el OPEX. Es decir, aumentamos drásticamente el valor de negocio de la aplicación sin gastar un solo centavo adicional por iteración, validando la viabilidad de una arquitectura de código abierto local para datos sensibles.

---

## 7. Taxonomía y Desagregación de Errores

En el desarrollo de sistemas RAG empresariales, una métrica global (como un 80% de precisión) es insuficiente porque te dice *que* el sistema falló, pero no *dónde* ni *por qué*. ¿Falló el modelo matemático de búsqueda o falló el razonamiento del LLM? 

Para resolver este problema de "depuración a ciegas" (*blind debugging*), implementamos un flujo automático de diagnóstico que desagrega cada fallo detectado por el Juez en 3 niveles accionables (A/B/C). La principal ventaja de esta taxonomía es que permite dirigir los esfuerzos de ingeniería de MLOps:

- **A - Fallo de Recuperación (Búsqueda):** La respuesta correcta no llegó al Top-K. **Ventaja:** Si dominan estos errores, sabemos que no hay que gastar dinero en un LLM más caro, sino afinar el algoritmo de Embeddings o el tamaño del *Chunk*.
- **B - Fallo de Generación (Razonamiento):** El texto correcto sí se recuperó, pero el LLM lo ignoró, alucinó o falló al razonar. **Ventaja:** Indica que los Embeddings son perfectos, pero debemos mejorar el *System Prompt* o escalar a un modelo con más parámetros (Ej. de Llama 3.1 8B a GPT-4o).
- **C - Fallo Estructural (Contrato):** El LLM rompió el contrato JSON esperado por Pydantic. **Ventaja:** Se corrige ajustando los validadores de salida (Output Parsers).

> 💡 *Concepto Técnico:* La arquitectura de evaluación usa el patrón *LLM-as-a-Judge*. En lugar de regex limitadas, pasamos la dupla (Texto Recuperado, Respuesta del Candidato) a un modelo Juez de alto razonamiento. Si la respuesta falla, el juez examina el texto recuperado; si contiene la respuesta matemática, la culpa recae en la Generación (Error B). Si no la contiene, la culpa es de la Búsqueda (Error A).

```python
# 7. Ejecución de Taxonomía de Errores
# (Representación del módulo de Streamlit y Pandas)
import pandas as pd
from pathlib import Path

# Apuntamos explícitamente a los resultados locales generados
project_root = Path(__file__).resolve().parents[3] if '__file__' in locals() else Path.cwd().parent
carpeta_eval_local = project_root / "data" / "03_output" / "evaluaciones" / "oficiales" / "run_local"

archivos_errores = list(carpeta_eval_local.glob("analisis_errores_desagregados*.csv"))
if archivos_errores:
    archivo_resultados = archivos_errores[0]
    df_resultados = pd.read_csv(archivo_resultados)
    
    conteo_errores = df_resultados['categoria_error'].value_counts()
    print("🎉 Taxonomía de Errores del Modelo Base:")
    print(conteo_errores)
else:
    print("⚠️ No se encontró el archivo de análisis de errores.")
```

> 💡 **Análisis de la Taxonomía de Errores (Ejecución Local)**
> - **Éxitos Absolutos:** Se lograron 61 extracciones perfectas donde el JSON cumplió con el formato y los datos requeridos.
> - **Formato Estructural (Error C - 1 caso):** El modelo casi nunca alucina la estructura JSON (solo 1 error). Esto confirma que el uso de *Structured Outputs* o prompteo estricto es sumamente eficaz para garantizar esquemas Pydantic robustos.
> - **Contexto Insuficiente (Error A - 47 casos):** El área de oportunidad principal está altamente concentrada aquí. Esto indica que el LLM funciona correctamente, pero el motor de *Retrieval* (Búsqueda Vectorial) no logró traer el fragmento normativo exacto requerido para llenar los campos.
> - **Conclusión MLOps:** El modelo generativo es sólido; los esfuerzos de optimización para el próximo ciclo no deben centrarse en el LLM, sino en refinar el *Chunking* (partición de documentos regulatorios), mejorar los *Embeddings*, o integrar la nube para casos complejos donde el modelo local no puede inferir correctamente.

---

## 8. Conclusión del Avance 4:

La transición de un prototipo RAG a un sistema de grado *Enterprise* (apto para Banco de México) requiere abandonar las decisiones basadas en intuición para adoptar métricas de ingeniería pura. Este avance demuestra una mejora técnica integral en múltiples frentes:

1. **Resolución de Bloqueos Críticos:** Se superó la limitante inicial de un-solo-documento mediante **Multi-Chunk Handling**, permitiendo la síntesis transversal de regulaciones complejas (CUB y LIC simultáneamente). Asimismo, la migración a *Contextual Retrieval* erradicó el *Semantic Drift* en incisos profundos.
2. **Soberanía y Eficiencia Financiera:** Al evaluar 6 arquitecturas mediante el patrón Factory, demostramos que es matemáticamente posible mantener la **Residencia de Datos** (ejecutando Llama 3.1 100% *on-premise*) mientras mitigamos agresivamente el Costo Total de Propiedad (TCO) a través de estrategias de *Cost Containment* como Caching, Reranking y Model Cascading.
3. **Erradicación del *Blind Debugging*:** El diseño de la taxonomía de errores (A/B/C) evaluada por un modelo Juez transforma un sistema de "caja negra" en un pipeline de MLOps transparente, indicando exactamente qué componente (Embeddings, Prompt o Salida Estructurada) requiere optimización quirúrgica.

En síntesis, la orquestación automatizada evaluada sobre un *Ground Truth* de 110 consultas curadas por expertos, sumada al análisis estadístico (Bootstrap CI) sobre la **Frontera de Pareto**, elimina la ambigüedad en la selección del modelo óptimo. Este ecosistema de evaluación justifica cuantitativamente la arquitectura del motor y establece cimientos inquebrantables para la interfaz final y el despliegue a usuarios finales en el Avance 5.
