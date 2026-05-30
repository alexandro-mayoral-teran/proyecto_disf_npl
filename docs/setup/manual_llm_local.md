# Manual de Ejecución Local: Ollama y vLLM

Este manual detalla cómo inicializar y conectar motores de inferencia locales (Ollama y vLLM) al pipeline del proyecto DISF. Dado que ambos exponen una API compatible con OpenAI de forma nativa, la integración con nuestro código es completamente transparente gracias a la configuración implementada en el archivo `.env`.

> **Nota:**
> De acuerdo a la tabla comparativa del proyecto:
> - **Ollama** (Formato GGUF): Destaca por su facilidad de configuración (un binario). Es ideal para **desarrollo local** y pruebas en hardware limitado (Laptops, PCs de escritorio).
> - **vLLM** (Formato safetensors): Ofrece un *throughput* alto gracias a su manejo de memoria (PagedAttention) y excelente soporte Multi-GPU. Es el estándar de la industria para **producción general**.

---

## 1. Configuración e Inicialización de Ollama

Ollama es la forma más rápida de tener un LLM corriendo localmente, empaquetado en un solo binario.

### Instalación
Descarga e instala Ollama desde su sitio web oficial: [ollama.com](https://ollama.com). 
- **Linux:** `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows / Mac:** Descarga el instalador gráfico y ejecútalo.

### Inicialización
Abre tu terminal y ejecuta el siguiente comando para descargar y correr un modelo. En este ejemplo, usaremos Llama 3.1 (8 billones de parámetros):
```bash
ollama run llama3.1:8b
```
> **Tip:** Al momento de ejecutar este comando, Ollama descargará los pesos y levantará automáticamente un servidor en segundo plano compatible con OpenAI en el puerto `11434`.

---

## 2. Configuración e Inicialización de vLLM

vLLM está diseñado para maximizar el rendimiento en tarjetas gráficas (GPUs) dedicadas (NVIDIA o AMD), ideal para cuando decidas pasar el proyecto DISF a un entorno de producción o a un servidor robusto.

### Instalación
Se recomienda usar un entorno virtual de Python. vLLM se instala fácilmente vía `pip`:
```bash
pip install vllm
```

### Inicialización
vLLM incluye un servidor web integrado que emula la API de OpenAI. Necesitas especificar el modelo (se descargará directamente desde HuggingFace, por ejemplo `meta-llama/Meta-Llama-3.1-8B-Instruct`).

Ejecuta en tu terminal:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000
```
> **Tip:** A diferencia de Ollama, vLLM por defecto levanta su servidor compatible con OpenAI en el puerto `8000`. Además, si tu modelo requiere autenticación en HuggingFace (como los de Meta), asegúrate de haber hecho login con `huggingface-cli login` antes.

---

## 3. ¿Cómo ligar estos motores al Proyecto DISF?

Gracias a la refactorización reciente (`config_llm.py`), conectar cualquiera de estos dos motores al pipeline (para RAG y Extracción) es tan sencillo como modificar tu archivo `.env`. ¡No necesitas tocar nada del código en Python!

### Escenario A: Conectando Ollama (Desarrollo)
Modifica tu `.env` de la siguiente manera:
```env
USE_LOCAL_LLM=true
# Puerto por defecto de Ollama
LOCAL_LLM_URL=http://localhost:11434/v1

# Nombre EXACTO del modelo que descargaste en Ollama (ej. llama3.1:8b)
LLM_MODEL_EXTRACTION=llama3.1:8b
LLM_MODEL_QA=llama3.1:8b
```

### Escenario B: Conectando vLLM (Producción)
Modifica tu `.env` de la siguiente manera:
```env
USE_LOCAL_LLM=true
# Puerto por defecto de vLLM
LOCAL_LLM_URL=http://localhost:8000/v1

# Nombre EXACTO del modelo de HuggingFace que vLLM está sirviendo
LLM_MODEL_EXTRACTION=meta-llama/Meta-Llama-3.1-8B-Instruct
LLM_MODEL_QA=meta-llama/Meta-Llama-3.1-8B-Instruct
```

> **Importante:**
> **Compatibilidad con Extracción JSON (Pydantic):** El pipeline de la DISF usa Pydantic (`client.beta.chat.completions.parse()`) para forzar la salida de las reglas de negocio en un formato JSON estructurado. 
> - **Ollama** soporta el formato de salida estructurada de OpenAI desde sus versiones recientes.
> - **vLLM** soporta *Guided Decoding* (decodificación guiada por JSON Schema). 
> 
> Si notas que el LLM falla al entregar la estructura correcta, asegúrate de estar utilizando la versión más actualizada de ambos motores.

---

## 4. Ejecución del Pipeline RAG (Pruebas Locales vs API)

La mayor ventaja de esta integración es que **la ejecución del proyecto es exactamente la misma** sin importar si estás usando OpenAI (API en la nube) o un modelo local (Ollama/vLLM). El RAG es completamente agnóstico al motor subyacente.

Una vez que hayas guardado tu configuración en el archivo `.env`, sigue estos pasos para probar el sistema:

### Paso 1: Levantar el Servidor Backend (FastAPI)
Abre una terminal en la raíz del proyecto y ejecuta la API principal:
```bash
# Asegúrate de tener tu entorno virtual activado
uvicorn main:app --reload
# (O si tienes un script main.py configurado: python main.py)
```

### Paso 2: Ejecutar el RAG

**Opción A: Desde la Interfaz Visual (Frontend)**
1. Abre tu navegador y dirígete a `http://localhost:8000` (o el puerto donde esté corriendo tu frontend).
2. Ve a la pestaña **Asistente Normativo** o al **Extractor de Formularios**.
3. Haz una consulta (ej. *"¿Cuáles son las reglas para reportar fideicomisos?"*). 
4. Si configuraste `USE_LOCAL_LLM=true`, notarás en la consola de tu servidor Ollama/vLLM que la petición se está procesando localmente.

**Opción B: Mediante cURL o Postman**
Puedes probar directamente el endpoint RAG del agente. Ejemplo usando cURL:
```bash
curl -X 'POST' \
  'http://localhost:8000/api/rag/qa' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "¿Qué es la DISF y cuáles son sus facultades?",
  "top_k": 3
}'
```

### Consideraciones Finales de Rendimiento
- **Velocidad:** OpenAI (`gpt-4o`) siempre responderá más rápido que Ollama corriendo en tu CPU/Laptop. Si usas vLLM con una buena GPU, la velocidad puede igualar o superar a OpenAI.
- **Calidad de Extracción:** Modelos grandes (ej. 70B o 405B) siempre extraerán estructuras JSON complejas mejor que modelos pequeños (8B). Si notas que Llama 3 8B alucina campos en la extracción Pydantic, considera subir a un modelo local más grande o usar OpenAI temporalmente para esa tarea específica (poniendo `USE_LOCAL_LLM=false`).
