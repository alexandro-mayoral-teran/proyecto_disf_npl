# 🏦 Diseño Automatizado de Requerimientos de Información Financiera con NLP

**Proyecto Integrador de Maestría**  
**Dominio:** Banco de México - Dirección de Información del Sistema Financiero (DISF)

---

## 📌 Descripción Ejecutiva del Proyecto

### Contexto y Problemática
De acuerdo con la Ley del Banco de México, una de las funciones prioritarias de la institución es procurar la estabilidad y el sano desarrollo del sistema financiero. Para lograrlo, la Dirección de Información del Sistema Financiero (DISF) recaba y analiza información mediante "requerimientos de información" (formularios y catálogos) diseñados con base en vasta y compleja documentación regulatoria (ej. la Circular Única de Bancos).

Actualmente, el proceso de traducir disposiciones normativas en estructuras de datos tabulares es una tarea manual, altamente especializada e intensiva en tiempo, lo que puede derivar en inconsistencias técnicas y operativas entre diferentes equipos.

### Propuesta de Solución
Este proyecto propone el desarrollo de una herramienta basada en Procesamiento de Lenguaje Natural (NLP) y Modelos de Lenguaje Grande (LLMs). El sistema actúa como un **"Especialista Digital Regulador"**, capaz de ingerir documentos legales complejos e identificar automáticamente variables de información relevantes. 

Su núcleo funcional se basa en arquitecturas de datos estrictas (mediante `Pydantic`) para asegurar que el modelo estructure el texto en el formato exacto requerido por los sistemas de bases de datos del Banco (ej. SAIFWeb).

### Valor Institucional
Esta iniciativa representa un primer paso hacia la adopción de Inteligencia Artificial para la automatización de procesos críticos de supervisión. La herramienta **no busca reemplazar el juicio del experto financiero, sino empoderarlo**, reduciendo drásticamente el tiempo de "traducción" de requerimientos y asegurando la estandarización y calidad de los datos acopiados por el Banco Central.

---

## 🎯 Alcance y Entregables (Producto Mínimo Viable)

Para garantizar su viabilidad en un entorno operativo, el proyecto delimita su alcance inicial a la generación de los siguientes artefactos a partir del texto normativo:

1. **Estructura tabular de los formularios:** Nombres de campo, tipos de dato, longitudes y descripciones funcionales.
2. **Propuesta de catálogos asociados:** Identificación de variables que requieren listas cerradas de valores.
3. **Auditoría y Mejoras (Opcional):** Detección de ambigüedades en el texto y sugerencias proactivas de validaciones de negocio.

---

## 📊 Metodología de Evaluación y Framework de Pruebas

Para asegurar la viabilidad institucional y evaluar cada etapa del flujo (Retrieval + Generación), se empleará un marco de evaluación riguroso:

**1. Evaluación de Recuperación de Información (RAG Retrieval):**
Previo a la generación, se evalúa la capacidad de los modelos de búsqueda para encontrar los documentos correctos.
*   **Golden Dataset de IR (Ground Truth):** Construcción (antes de realizar pruebas) de un benchmark de 20-30 consultas representativas, con 3-5 fragmentos regulatorios relevantes etiquetados por expertos.
*   **Métricas Primarias (A priori):** La métrica principal será **Recall@5**, junto con **nDCG@10**.
*   **Comparativa Estricta:** El desempeño del pipeline híbrido (con Cross-Encoder Reranking y HyDE) se medirá contra la línea base (Floor) que representa **BM25**, para comprobar empíricamente su "lift" y valor añadido.

**2. Evaluación de Modelos Generativos (LLMs):**
Una vez recuperado el contexto, se evalúa la salida final estructurada y la idoneidad del sistema completo.
*   **Telemetría y Costos (ROI):** Rastreo de latencia, costo en USD y estimación del "Costo de revisión humana a escala" (minutos por documento analizado) vs el proceso manual.
*   **Análisis de Errores por Etapa:** Distinción clara para detectar si el error ocurrió por (a) Fallo en Retrieval, (b) Alucinación Generativa o (c) Error de formato estructurado.
*   **Data Contamination Check:** Pruebas "sin contexto" (*no-context test*) para descartar que el LLM simplemente memorizó las normativas en su pre-entrenamiento.

---

## 🏗️ Arquitectura del Sistema

El proyecto sigue una arquitectura modular "API-First" para separar la lógica de extracción de datos, el procesamiento inteligente y la interfaz de usuario:

*   **`data/`**: Contiene el *Golden Dataset*. Los documentos originales en PDF/Excel se transforman a un formato Markdown (`.md`) estructural para optimizar la ventana de contexto del LLM y preservar las tablas normativas.
*   **`src/`**: Motor central del proyecto (Backend).
    *   `ingesta/`: Parsers y conversores bajo la clase `IngestorDocumentos` (Excel/PDF a Markdown estructurado).
    *   `nlp_core/`: Agentes extractores (Pydantic), `RegulacionChunker` (Patrón Strategy), `MotorVectorizacion` (ChromaDB) y `MotorBusqueda` (Búsqueda Híbrida y Multi-Query).
*   **`api/` (Planeado):** Capa de servicios usando **FastAPI** para exponer el motor de NLP a futuras interfaces.
*   **`app/` (Planeado):** Interfaz de usuario interactiva y ligera (MVP) construida con **Streamlit**.
*   **`notebooks/`**: Entorno de experimentación, reportes de avances (EDA) y validación de métricas.

## 🛠️ Stack Tecnológico

*   **Lenguaje:** Python 3.10+
*   **Procesamiento de Datos:** Pandas, Tabulate, Expresiones Regulares (`re`).
*   **Contrato de Datos y Validación:** Pydantic.
*   **Orquestación y API:** FastAPI (Próximamente).
*   **Frontend UI:** Streamlit (Próximamente).

## 🚀 Mapa de Ruta (Roadmap de 10 Semanas)

| Fase | Estado | Descripción Técnica |
| :--- | :--- | :--- |
| **1. Cimentación y Dataset** | 🟢 Completado | Extracción de CUB y catálogos de Excel a Markdown. Refactorización en Arquitectura Orientada a Objetos (OOP). |
| **2. RAG Engine y Retrieval** | 🟡 En progreso | Implementación de Chunking dinámico, Búsqueda Híbrida (RRF), **HyDE**, Query Expansion y **Cross-Encoder Reranking**. |
| **3. Evaluación Cuantitativa (Arena)** | 🟡 En progreso | Construcción del Ground Truth (20-30 queries). Benchmarking sistemático (Recall@5) comparando baselines (BM25) vs SOTA. |
| **4. Integración y API** | ⚪ Pendiente | Empaquetado del pipeline de `src/` mediante endpoints de FastAPI. |
| **5. Interfaz y Entrega** | ⚪ Pendiente | Construcción del MVP web en Streamlit, pruebas de consistencia final y documentación académica. |

## ⚙️ Configuración del Entorno (Desarrollo Local)

1. Clonar el repositorio.
2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instalar dependencias *(Archivo requirements.txt en construcción)*:
   ```bash
   pip install -r requirements.txt
   ```