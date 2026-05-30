# Guía de Construcción del Notebook: Frontera de Pareto y Evaluación Final (MA1 y MA7)

Este documento sirve como plano arquitectónico para crear el gran Jupyter Notebook (`04_evaluacion_pareto.ipynb`). La finalidad del notebook es consumir los módulos del Laboratorio (`src/lab/`), evaluar múltiples configuraciones de LLMs/Retrievers, calcular el Costo Total de Propiedad (TCO) y generar la Frontera de Pareto final que soportará la decisión Go/No-Go para Banxico.

---

## 🏗️ Arquitectura del Notebook de Pareto

El notebook deberá estar dividido en **6 etapas clave**:
1. **Configuración de Entorno e Importaciones:** Carga de variables de entorno y los módulos utilitarios (`RastreadorTelemetria`, `plot_frontera_pareto`).
2. **Carga del Dataset Extendido:** Consumo de `data/evaluacion_dataset.json` (ahora con 110 consultas).
3. **Definición de Candidatos (MA1):** Configurar el bucle para iterar sobre las 6 estrategias (ej. Baseline Léxico, Híbrido + GPT-4o-mini, Híbrido + Llama3 Local, etc).
4. **Ejecución del RAG y Telemetría:** Correr las consultas, inyectar el contexto y rastrear tiempos/tokens exactos de cada modelo.
5. **Generación del Gráfico de Pareto (MA7 y ENS-C):** Llamar a `plot_frontera_pareto` con los resultados consolidados.
6. **Conclusión y Decisión (DEP-A):** Redactar un párrafo justificando qué modelo se elegirá para Producción.

---

## 📝 Referencia para Playground Local (Opcional)

Si además de la evaluación masiva deseas un playground interactivo, puedes seguir la siguiente estructura original:

## Estructura Propuesta para el Notebook

El notebook deberá estar dividido en **6 etapas clave**:
1. **Configuración de Entorno e Importaciones:** Carga de variables de entorno, configuración de rutas y verificación de dependencias.
2. **Pruebas de Conectividad Híbrida:** Diagnóstico visual rápido de la nube (OpenAI) y el servidor local (Ollama).
3. **Carga y Fragmentación con Inyección de Metadatos:** Simulación de cómo se leen las normativas, se fragmentan por encabezados y se les inyecta contexto.
4. **Consulta Híbrida y Semántica (Retrieval):** Búsqueda en la base de datos ChromaDB usando similitud de embeddings y BM25 léxico.
5. **Generación Local con Llama 3.1 (RAG Extremo a Extremo):** Generación de respuestas normativas en tu GPU RTX 4060 en menos de un segundo.
6. **Módulo de Telemetría (Velocidad e Inferencia):** Medición de latencia y velocidad de generación (tokens/segundo).

---

## Código y Narrativa para cada Celda (Listo para copiar)

A continuación se detalla el contenido exacto (Markdown y Código Python) que debe tener cada celda del notebook:

### Celda 1: Títulos y Contexto (Markdown)
```markdown
# Playground RAG Híbrido: Inferencia Local (Ollama) vs Nube (OpenAI)

Este notebook permite interactuar de forma aislada con los componentes del proyecto DISF. 
Aquí probaremos cómo se comporta el modelo local **Llama 3.1 8B** corriendo en nuestra GPU RTX 4060 frente a la API de **OpenAI**, comparando su precisión, velocidad de respuesta y telemetría.
```

### Celda 2: Carga del Entorno e Imports (Código)
```python
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# 1. Configurar rutas para importar desde 'src' robustamente
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == "notebooks" else notebook_dir
sys.path.append(str(project_root))

# 2. Cargar variables del archivo .env
load_dotenv(project_root / ".env")

print("✅ Entorno cargado exitosamente.")
print(f"Ruta raíz del proyecto: {project_root}")
print(f"¿Usar LLM Local para QA?: {os.getenv('USE_LOCAL_QA')}")
print(f"Modelo local para QA: {os.getenv('LLM_MODEL_QA', 'llama3.1:8b')}")
```

### Celda 3: Verificación de Conexión de Modelos (Código)
```python
from src.nlp_core.config_llm import get_llm_client, get_langchain_chat

print("--- Diagnóstico de Modelos ---")

# Probar OpenAI (Nube)
try:
    openai_client = get_llm_client(task="extraction") # Fuerza nube según configuración estándar
    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Di 'Nube OK'"}],
        max_tokens=10
    )
    print(f"☁️ OpenAI Nube: {res.choices[0].message.content.strip()}")
except Exception as e:
    print(f"❌ Error en OpenAI: {e}")

# Probar Ollama (Local)
try:
    local_chat = get_langchain_chat(task="qa") # Utiliza configuración del .env
    res = local_chat.invoke("Di 'Ollama Local OK' en una palabra")
    print(f"💻 Ollama Local (RTX 4060): {res.content.strip()}")
except Exception as e:
    print(f"❌ Error en Ollama: {e}")
```

### Celda 4: Inicialización del Motor de Búsqueda (Código)
```python
from src.nlp_core.vectorizacion import MotorVectorizacion
from src.nlp_core.retrieval import MotorBusqueda
from src.nlp_core.pipeline import PipelineRecuperacion

# Configurar rutas del almacenamiento de vectores
chroma_path = project_root / "data" / "02_vectorstore" / "chroma_db"

print(f"Conectando a base de datos vectorial en: {chroma_path}...")
motor_vectorizacion = MotorVectorizacion(persist_directory=str(chroma_path))
motor_busqueda = MotorBusqueda(motor_vectorizacion)

# Crear el pipeline de búsqueda avanzado
pipeline = PipelineRecuperacion(motor_busqueda)
# Activamos búsqueda híbrida RRF para máxima precisión
pipeline.configurar(
    usar_expansion=False, 
    usar_busqueda_hibrida=True, 
    usar_cross_encoder=False
)

print("✅ Motor de búsqueda e indexador ChromaDB inicializados.")
```

### Celda 5: Pruebas de Búsqueda e Inyección de Metadatos (Código)
```python
# Definimos una consulta típica del sector financiero
query_prueba = "¿Cuáles son los requisitos de información para los créditos de auto?"

print(f"🔍 Ejecutando Búsqueda Híbrida para: '{query_prueba}'\n")
documentos_recuperados = pipeline.buscar(query_prueba, k=3)

for idx, doc in enumerate(documentos_recuperados, 1):
    print(f"--- Fragmento #{idx} ---")
    print(f"Metadatos: {doc.metadata}")
    print(f"Contenido preliminar: {doc.page_content[:250]}...")
    print("-" * 30 + "\n")
```

### Celda 6: Generación RAG Extremo a Extremo (Código)
```python
# Crear contexto unificado a partir de los fragmentos recuperados
contexto_unificado = "\n\n".join([doc.page_content for doc in documentos_recuperados])

# Definir el Prompt del Sistema para el Asistente Regulador
prompt_sistema = (
    "Eres un especialista digital regulador para la DISF de Banxico.\n"
    "Responde a la pregunta del analista financiero basándote estrictamente en el "
    "contexto normativo proporcionado a continuación. Si el contexto no contiene la "
    "información, sé honesto e indica que no cuentas con los datos suficientes.\n\n"
    f"CONTEXTO NORMATIVO:\n{contexto_unificado}"
)

# Ejecutar el RAG usando el cliente configurado en el .env (Local en RTX 4060 o Nube)
print("🧠 Generando respuesta regulatoria...")
t_inicio = time.time()

llm = get_langchain_chat(task="qa", temperature=0.2)
respuesta_completa = llm.invoke([
    {"role": "system", "content": prompt_sistema},
    {"role": "user", "content": query_prueba}
])

t_fin = time.time()
duracion = t_fin - t_inicio

print(f"\n⏱️ Inferencia completada en {duracion:.2f} segundos.\n")
print("================= RESPUESTA FINAL =================\n")
print(respuesta_completa.content)
print("\n===================================================")
```

### Celda 7: Módulo de Telemetría Detallada (Código)
```python
import tiktoken

# Contar tokens del prompt y de la salida
try:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_prompt = len(encoding.encode(prompt_sistema))
    tokens_respuesta = len(encoding.encode(respuesta_completa.content))
    total_tokens = tokens_prompt + tokens_respuesta
    
    tokens_por_segundo = tokens_respuesta / duracion if duracion > 0 else 0
    
    print("📊 Métricas de Inferencia Local:")
    print(f" - Tokens del Prompt (Entrada): {tokens_prompt}")
    print(f" - Tokens de la Respuesta (Salida): {tokens_respuesta}")
    print(f" - Tokens Totales Procesados: {total_tokens}")
    print(f" - Velocidad de Generación: {tokens_por_segundo:.2f} tokens/segundo")
    print(f" - Dispositivo de ejecución estimado: GPU NVIDIA RTX 4060")
except Exception as e:
    print(f"No se pudieron calcular las métricas de tokens detalladas: {e}")
```

---

## Instrucciones para Crear el Notebook Físicamente en VS Code

Cuando estés listo para materializar este plano en un archivo `.ipynb`:
1. Crea un nuevo archivo en VS Code bajo la ruta `notebooks/playground_local.ipynb`.
2. VS Code te preguntará qué Kernel usar. Selecciona el intérprete de Python de tu entorno virtual (`.venv`).
3. Crea celdas de tipo "Markdown" para los títulos explicativos y celdas de "Código" para los scripts de Python.
4. Ejecuta celda por celda y disfruta de la velocidad de inferencia de tu GPU local RTX 4060.
