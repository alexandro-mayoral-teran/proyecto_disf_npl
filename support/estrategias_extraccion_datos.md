# Alternativas de Extracción de Información Estructurada (LLMs)

Este documento detalla las estrategias evaluadas y propuestas para la extracción de formularios, catálogos y reglas de negocio a partir de documentos normativos (ej. circulares de Banxico/CNBV) para el proyecto DISF.

## 1. Full Context Prompting (Línea Base Actual)
**Mecanismo:** El documento completo (previamente parseado a Markdown conservando sus tablas y jerarquías) se inyecta íntegramente en el prompt del LLM (ej. `gpt-4o`), forzando la salida a través de `Structured Outputs` (Pydantic).
*   **Pros:** 
    *   Sencillez de implementación.
    *   Excelente para correlacionar datos dispersos porque el modelo tiene todo el contexto en su "memoria" simultáneamente.
*   **Contras:**
    *   Costo muy elevado en producción para normativas extensas.
    *   Riesgo de fenómeno *"Lost in the Middle"* (el LLM tiende a ignorar instrucciones o datos ubicados en el medio de textos muy largos).
    *   Limitado estrictamente por el tamaño de la ventana de contexto (ej. 128k tokens).

## 2. Agentic RAG Extractivo (Recuperación Vectorial)
**Mecanismo:** El Markdown se divide en fragmentos semánticos (usando *MarkdownHeaderTextSplitter* para no romper artículos) y se almacena en una VectorDB (ej. Chroma, pgvector). El agente realiza consultas específicas para poblar el esquema JSON poco a poco.
*   **Pros:** 
    *   Altamente escalable y económico; el modelo solo lee los fragmentos relevantes.
    *   Evita por completo el límite de tokens.
*   **Contras:**
    *   Riesgo de omisión: Si la similitud coseno (búsqueda vectorial) falla en encontrar un artículo, ese campo quedará fuera del JSON final.

## 3. Enfoque "Map-Reduce" (Extracción Exhaustiva por Lotes)
**Mecanismo:** El documento se divide en chunks jerárquicos (ej. por Artículos o Capítulos). 
1.  **Map:** Se procesa cada chunk individualmente (puede ser en paralelo), pidiendo al LLM que extraiga un "mini-JSON" de reglas y catálogos exclusivos de esa sección.
2.  **Reduce:** Un LLM final (el "Ensamblador") consolida todos los mini-JSONs, fusiona duplicados, normaliza nombres y genera el esquema Pydantic definitivo.
*   **Pros:** 
    *   Garantiza **100% de cobertura** del texto. Ideal para casos de uso regulatorio donde saltarse una sola regla o validación es inaceptable.
    *   Evita el "Lost in the Middle" al forzar la atención en fragmentos pequeños.
*   **Contras:**
    *   Mayor consumo de llamadas a la API que el enfoque RAG, aunque generalmente es más barato que procesar todo el documento múltiple veces.

## 4. Flujo Multi-Agente Especializado (Agentic Workflows)
**Mecanismo:** Se utilizan frameworks (como LangGraph, AutoGen o CrewAI) para dividir la carga cognitiva en una "fábrica" de agentes especializados.
*   🤖 **Agente Matemático:** Especialista exclusivo en extraer y validar fórmulas.
*   🤖 **Agente de Catálogos:** Especialista exclusivo en detectar listas cerradas y posibles valores enumerados.
*   🤖 **Agente Revisor (Sintetizador):** Audita el trabajo de los anteriores y ensambla el modelo `RequerimientoInformacion`.
*   **Pros:** 
    *   Aumenta drásticamente la calidad y precisión de la extracción al aplicar el principio de "Divide y Vencerás".
    *   Permite delegar tareas simples a modelos más baratos (ej. `gpt-4o-mini`) y dejar el ensamblaje al modelo potente.

## 5. GraphRAG (Grafos de Conocimiento)
**Mecanismo:** En lugar de indexar trozos de texto, el LLM procesa el documento extrayendo "Entidades" (ej. *Tasa de Interés*, *Cartera Vencida*) y "Relaciones" (ej. *Se calcula con*, *Aplica a*), construyendo un Grafo de Conocimiento (Knowledge Graph) en bases de datos como Neo4j.
*   **Pros:** 
    *   Es el estado del arte para normativas altamente interconectadas (*Multi-hop reasoning*). Si el Capítulo I define una variable que luego es alterada en el Capítulo V, GraphRAG mapea esa conexión de forma explícita y perfecta.
*   **Contras:**
    *   Curva de aprendizaje pronunciada y alta complejidad arquitectónica.

---
### 🎓 Sugerencia para la Evaluación del Proyecto Integrador
Para lograr una tesis de maestría robusta y alineada con los problemas de frontera en la industria FinTech/RegTech, la evaluación comparativa ideal sería:
**Línea Base (Full Context)** vs **RAG Extractivo** vs **Map-Reduce**. 

Esta comparativa demostrará un profundo entendimiento de los trade-offs entre costo, latencia, completitud y prevención de alucinaciones.
