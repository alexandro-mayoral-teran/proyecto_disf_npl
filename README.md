# 🏛️ ARIF: Asistente Regulatorio para Información Financiera

## 1. ¿Qué hace y qué problema resuelve?
**ARIF** es un asistente digital diseñado para la Dirección de Información del Sistema Financiero (DISF) del Banco de México. Ingiere documentos legales densos (como la Circular Única de Bancos) para responder consultas normativas con alta precisión y extraer automáticamente esquemas de formularios y metadatos.

**El Problema:** Analizar cientos de páginas de normatividad para diseñar requerimientos de información es un proceso manual y propenso a inconsistencias. ARIF automatiza esta "traducción" de texto legal a datos estructurados, operando bajo una política estricta de **Privacidad Local-First** para asegurar que la información sensible nunca salga de la institución sin validación.

## 2. Arquitectura del Sistema
El proyecto opera como un ecosistema modular de microservicios basado en **RAG (Retrieval-Augmented Generation) y Extracción Estructurada**.

* **Ingesta Inteligente:** Conversión de PDFs a Markdown preservando la jerarquía jurídica. Implementación de **Contextual Retrieval** asíncrono para inyectar memoria global a fragmentos aislados.
* **Recuperación (Retrieval):** Búsqueda híbrida que fusiona la precisión léxica (BM25) con el entendimiento semántico (Embeddings en ChromaDB) usando Reciprocal Rank Fusion (RRF) y reordenamiento con Cross-Encoders.
* **Generación (Model Cascading):** Ruteo dinámico donde consultas locales se resuelven de forma gratuita y segura con **Llama 3.1 8B**, y escalan a la Nube (GPT-4o) solo si el sistema autoevalúa que su nivel de fidelidad (*Faithfulness*) es bajo.
* **Extracción Long-Context:** Para la creación de formularios y manifiestos, el RAG se apaga. Se pasa el documento entero al modelo forzándolo a devolver esquemas validados con **Pydantic** a través de *Structured Outputs*, incluyendo la capacidad de **Extracción Guiada**.

## 3. ¿Cómo se evalúa? (Métricas del Laboratorio)
No dependemos de adjetivos; la calidad del sistema fue auditada mediante un pipeline inspirado en MLOps (RAGAS) sobre un *Golden Dataset* de 109 consultas reales:

* **Recall@10:** **73.33%** en modo base, y hasta **90.0%** activando el pipeline completo (Multi-Query + HyDE + Cross-Encoder).
* **Latencia Operativa:** Consultas frecuentes resueltas en **< 0.5s** gracias al Caché Semántico Híbrido operado por un Juez LLM local.
* **Calibración:** Medición de *Expected Calibration Error (ECE)* y *Faithfulness* validada mediante un modelo cruzado (GPT-4o) actuando como juez objetivo para castigar el sesgo de benevolencia del modelo local.

## 4. Limitaciones Actuales
Siendo ingenieros, transparentamos los límites de la arquitectura en su versión actual:

* **Lost in the Middle:** El enfoque de Extracción *Long-Context* sufre degradación de memoria al procesar regulaciones que superen los 100,000 tokens de golpe, pudiendo omitir campos profundos del formulario.
* **Restricciones de Hardware:** El modelo local Llama 3.1 8B requiere estrictamente GPUs con al menos 8GB de VRAM compartida. Ejecutar el orquestador en CPU degrada la latencia a niveles inoperables para un caso de uso interactivo.
* **Prompt Injection en Extracción Guiada:** La nueva función que permite a los usuarios inyectar instrucciones directas ("extrae solo esto...") carece actualmente de un filtro de toxicidad avanzado para ese campo de texto libre, requiriendo un *Guardrail* explícito antes de producción.

---

## 📚 Documentación Oficial

Toda la ingeniería y decisiones de diseño están respaldadas por documentación formal para facilitar el "Hand-off" institucional.

**Arquitectura y Diseño:**
* 📘 [Estrategia RAG y Modelos](docs/arquitectura/estrategia_rag.md): Justificación formal de la búsqueda híbrida, métricas de evaluación y uso de Llama 3.1.
* 📘 [Extracción y Gobernanza de Datos](docs/arquitectura/extraccion_y_gobernanza_datos.md): Documentación del flujo Long-Context para formularios y metadatos.
* 📘 [Estrategias de Chunking Avanzado](docs/arquitectura/estrategias_chunking_avanzado.md): Detalles sobre fragmentación y *Contextual Retrieval*.
* 🗺️ [Hub de Diagramas RAG](docs/arquitectura/arquitectura_rag.md) (y sus sub-diagramas Mermaid en la misma carpeta).

**Laboratorio y MLOps:**
* 🧪 [Manual de Ejecución de Pruebas](docs/evaluaciones/manual_ejecucion_pruebas.md): Cómo correr simulaciones automatizadas *Config-Driven*.
* 🧪 [Guía de Interpretación de Resultados](docs/evaluaciones/guia_interpretacion_resultados.md): Entendiendo NDCG, Recall y la taxonomía de fallos.
* 🧪 [Manual de Laboratorio y Métricas](docs/evaluaciones/manual_laboratorio_metricas.md).

**Infraestructura:**
* 🛠️ [Manual de Despliegue en Producción](docs/setup/deploy_produccion.md): Consideraciones FinOps, TCO y Arquitectura Cloud (GCP/AWS).
* 🛠️ [Manual de LLMs Locales](docs/setup/manual_llm_local.md): Setup de Ollama y vLLM.

---

## 🚀 Mapa de Ruta y Próximos Pasos

El proyecto demostró la viabilidad técnica del modelo. Los siguientes pasos para un despliegue piloto son:

1. **Hardware Profiling e Infraestructura:** Migrar de `Ollama` a `vLLM` en servidores internos del banco equipados con aceleración GPU (ej. NVIDIA L4) para soportar procesamiento asíncrono y alta concurrencia de analistas de forma fluida.
2. **HitL (Human-in-the-Loop):** Construir una interfaz donde el LLM asigne "Banderas de Confianza" a los campos que extrae, obligando al usuario humano a validarlos antes de impactar las bases de datos de la DISF.
3. **Escalabilidad Documental (Map-Reduce):** Transicionar la extracción de megadocumentos normativos a un flujo Multi-Agente, particionando la ley en Capítulos procesados en paralelo para erradicar el problema de olvido ("Lost in the middle").
4. **Gobernanza Cloud y Auditoría:** Asegurar dictámenes de ciberseguridad sobre el componente de escalado en la nube (Router Cascade) y blindar la aplicación contra *Prompt Injections*.

---

## ⚙️ Configuración del Entorno (Desarrollo Local)

Para probar ARIF de forma local:

1. **Clonar el repositorio.**
2. **Crear y activar un entorno virtual (Python 3.10+):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. **Instalar las dependencias actualizadas:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configurar el entorno:**
   Copia el archivo `.env.example` a `.env` y añade tus credenciales (OpenAI API Key) y la configuración de modelos (ej. Llama 3.1 vía Ollama en `LOCAL_LLM_URL`).
5. **Iniciar el servidor backend (FastAPI):**
   ```bash
   uvicorn api.main_api:app --reload
   ```
6. **Abrir la Interfaz de Usuario:**
   Navega a `http://localhost:8000/app/index.html` para utilizar la aplicación web.
