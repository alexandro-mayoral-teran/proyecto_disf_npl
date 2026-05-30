markdown_content = """# Ejecución Local: Cloud vs OSS + Trade-offs

## 1. ¿Qué es un modelo "Open Source"?
Existen tres categorías principales de modelos de IA, y la distinción es fundamental tanto a nivel técnico como legal:

* **Propietarios (GPT-4o, Claude, Gemini):** Acceso solo vía API. No conocemos la arquitectura, los datos de entrenamiento ni los pesos.
* **Open Weight (Llama 3, Mistral, Qwen):** Publican los pesos para ejecutarlos, pero no los datos de entrenamiento completos. En la práctica, cuando la industria habla de "modelos OSS", casi siempre se refiere a estos.
* **Open Source real (OLMo, Pythia):** Publican todo: pesos, datos, código de entrenamiento y documentación.

> **Importante:** La distinción importa legalmente. Cada modelo tiene una licencia que define si puedes usarlo comercialmente, modificarlo o redistribuirlo. Siempre revisa la licencia (ej. Apache 2.0, Llama License, Gemma License) antes de integrar un modelo en producción.

### ¿Por qué importa el OSS para AI Engineers?
El uso de modelos abiertos ofrece ventajas estratégicas claras:
* **Privacidad y compliance:** Los datos nunca salen de tu infraestructura. Para sectores como salud, gobierno o finanzas, esto no es opcional.
* **Control total:** Permite hacer *fine-tuning* con tus propios datos y cuantizar al nivel que necesites, sin depender de actualizaciones de terceros.
* **Costos predecibles:** Pagas por la infraestructura (GPU), no por tokens. A alto volumen, el costo por token se desploma.
* **Riesgo:** Eres responsable del mantenimiento, actualizaciones de seguridad, escalamiento y monitoreo.

---

## 2. Parámetros y Memoria Requerida

### ¿Qué son los parámetros?
Un parámetro es un valor numérico que el modelo aprendió durante el entrenamiento. Por ejemplo, "Llama 3 70B" significa que tiene ~70 mil millones de valores numéricos.
* *Regla general:* Más parámetros = mayor capacidad, pero también más memoria, cómputo y latencia.

| Rango | Tamaño | Uso típico |
| :--- | :--- | :--- |
| **Pequeños** | 1B - 3B | Tareas específicas, edge, móvil |
| **Medianos** | 7B - 14B | *Workhorses* de producción |
| **Grandes** | 32B - 70B | Razonamiento complejo |
| **Masivos** | 100B+ | Múltiples GPUs requeridas |

### ¿Cómo se mide la memoria VRAM requerida?
**Fórmula base:** `Memoria = Parámetros × Bytes por parámetro`
* **FP32:** 4 bytes/param
* **FP16 / BF16:** 2 bytes/param

**Ejemplo (Llama 3 8B en FP16):** `8B × 2 = 16 GB` solo para los pesos.
*Nota:* Se debe sumar el KV-cache, activaciones y overhead (aprox. +10-20% adicional).

---

## 3. Cuantización: Comprimiendo Modelos
La cuantización reduce la precisión numérica de los pesos (ej. 16 bits → 8 bits → 4 bits) para ahorrar memoria.

| Precisión | Llama 3 8B (Memoria) | Formato común |
| :--- | :--- | :--- |
| **FP16** (16 bits) | ~16 GB | safetensors, HuggingFace |
| **INT8** (8 bits) | ~8 GB | BitsAndBytes, GPTQ |
| **Q4_K_M** (4 bits) | ~4.5 GB | GGUF (Ollama/llama.cpp) |
| **Q2_K** (2 bits) | ~2.5 GB | GGUF (calidad muy limitada) |

*Nota:* `Q4_K_M` y `Q5_K_M` ofrecen el mejor balance tamaño/calidad.

### Guía práctica de cuantización
* **Chat / generación de texto:** `Q4_K_M` es casi indistinguible de FP16.
* **Extracción estructurada (JSON) & Coding:** `Q4_K_M` es suficiente/funciona bien.
* **Razonamiento matemático:** Degradación notable en precisiones menores a Q5.
* **Desarrollo local:** Usa `Q4_K_M` por defecto. **Benchmarks:** Usa FP16.

---

## 4. El Ecosistema: HuggingFace
HuggingFace es el hub central y punto de partida para los modelos OSS. 

**¿Qué encontrarás?**
* *Model cards* con documentación y benchmarks.
* Múltiples formatos por modelo (safetensors, GGUF, GPTQ, AWQ).
* Datasets para fine-tuning y *Spaces* para demos interactivas.
* La librería `Transformers` para cargar/ejecutar modelos con pocas líneas.

**Cómo navegar eficientemente:**
* Filtra por tarea (`text-generation`, `text2text-generation`).
* Busca *publishers* confiables para cuantizaciones (ej. *unsloth, bartowski, mradermacher*).
* **Lee el Model Card como Engineer evaluando:**
    * **Arquitectura y tamaño:** Contexto máximo y tipo (define requisitos de hardware).
    * **Licencia:** Apache 2.0 (libre), Llama License, Gemma License.
    * **Benchmarks:** MMLU, HumanEval, GSM8K, IFEval.
    * **Formatos:** safetensors (GPU), GGUF (Ollama), GPTQ/AWQ (cuantizado GPU).

---

## 5. Motores de Inferencia y Serving

### 5.0 Extra: llama.cpp (La base del ecosistema local)
Proyecto fundacional de inferencia local de LLMs escrito en C/C++ por Georgi Gerganov. Es el motor detrás de Ollama y otras herramientas.
* **¿Por qué importa?** Hizo viable la inferencia de LLMs en hardware de consumidor e introdujo el formato GGUF.
* **Soporte HW:** CPU puro, NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal), NPUs.
* **Uso directo:** Solo cuando necesitas máximo control, hardware no-NVIDIA, embeber el modelo en una app C/C++ o tener la mínima dependencia posible.

### 5.1 Ollama: Inferencia local simplificada
Es el "Docker" de los modelos de lenguaje. Usa `llama.cpp` as backend y modelos en formato GGUF.
* **Ventajas:** Instalación trivial, catálogo preconfigurado, API compatible con OpenAI (`http://localhost:11434/v1/`).
* **✅ Úsalo cuando:** Desarrollo local, probar modelos rápido, prototipos, pipelines 100% offline.

**Setup en 3 comandos:**
```bash
# 1. Instalar Ollama (Linux/Mac/Windows)
curl -fsSL [https://ollama.ai/install.sh](https://ollama.ai/install.sh) | sh

# 2. Descargar un modelo
ollama pull llama3.1:8b

# 3. Verificar que funciona
ollama run llama3.1:8b "Hola, funciona?"

# Ver modelos descargados
ollama list

