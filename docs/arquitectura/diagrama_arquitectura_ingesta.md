# 📥 Arquitectura del Módulo de Ingesta (ETL Vectorial)

Este diagrama detalla cómo se procesan los documentos regulatorios "crudos" de Banxico (CUB, LIC, etc.) hasta convertirse en fragmentos matemáticos listos para ser buscados por el RAG.

---

## 1. Topología del Pipeline de Ingesta (Mermaid)

```mermaid
graph TD
    %% Fuentes de Datos
    A[("Documentos Banxico<br>(PDFs Normativos)")] -->|Lectura Cruda| B

    %% Procesamiento
    B["chunking.py<br>Procesamiento de Texto"] -->|Parser Especializado| C["Extracción Semántica<br>(Tablas, Artículos, Incisos)"]
    C -->|Particionado Recursivo| D["Fragmentos (Chunks)<br>+ Metadatos (Archivo, Página)"]

    %% Vectorización
    D --> E["vectorizacion.py<br>Generador de Embeddings"]
    E -->|Llama a la API| F(("Modelo de Embeddings<br>text-embedding-3-small"))
    F -->|Devuelve Vectores| G

    %% Almacenamiento
    G["Gestor ChromaDB"] -->|Inserta| H[("ChromaDB Vector Store<br>(Persistencia Local)")]

    %% Estilos
    style A fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white
    style B fill:#e67e22,stroke:#d35400,stroke-width:2px,color:white
    style E fill:#e67e22,stroke:#d35400,stroke-width:2px,color:white
    style H fill:#9b59b6,stroke:#8e44ad,stroke-width:4px,color:white
    style F fill:#34495e,stroke:#2c3e50,stroke-width:2px,color:white
```

---

## 2. Flujo Explicado

1. **Extracción (Extract):** Los documentos fuente (PDFs pesados como la CUB o la Ley de Instituciones de Crédito) son leídos por el script `chunking.py`. No se lee como texto plano, sino que se respeta la jerarquía legal (Capítulos, Artículos, Incisos) para no perder el contexto.
2. **Transformación (Transform):** Los textos gigantes se parten en pedazos pequeños (*chunks*). Se les inyectan **Metadatos**, es decir, el fragmento guarda la memoria de qué página y qué archivo vino. Esto es clave para el RAG Multi-Documento.
3. **Carga (Load):** `vectorizacion.py` toma cada fragmento y lo pasa por el modelo matemático (Ej. `text-embedding-3-small` de OpenAI). Esto convierte las palabras en una lista de números (vectores). Finalmente, se guardan en **ChromaDB**, que funcionará como nuestro "Cerebro de Memoria a Largo Plazo".
