# Estrategias Avanzadas de Chunking para RAG

Este documento detalla las estrategias de última generación (State of the Art) para el procesamiento y división (chunking) de documentos masivos (ej. 600+ páginas) en sistemas RAG, así como su impacto en el pipeline de recuperación (Retrieval).

## 1. Contextual Retrieval (con Optimización Asíncrona)

**Concepto:** 
En documentos muy grandes, un fragmento aislado a menudo pierde el sentido de a qué tema pertenece. Esta estrategia utiliza un LLM durante la ingesta para leer el documento completo (o sección grande) y generar 1 o 2 oraciones de contexto que se le "pegan" como prefijo al chunk antes de guardarlo en la base vectorial (ChromaDB).

**Ventajas:**
- Disminuye drásticamente los errores de recuperación donde palabras clave ambiguas (ej. "límite del 5%") recuperan secciones del anexo equivocado.
- Es completamente transparente para el motor de búsqueda.

**El cuello de botella actual:**
El bucle actual procesa esto de manera secuencial:
```python
for doc in chunks:
    # Llamada síncrona a GPT-4o-mini
    respuesta = llm.invoke(...)
```
Para 1,000 chunks, esto significa esperar 1,000 respuestas una por una (aproximadamente 5 horas).

**Implementación (Optimización):**
Para reducir el tiempo a minutos, se debe usar **procesamiento asíncrono** (`asyncio.gather` en Python) o la **Batch API de OpenAI**. 
```python
import asyncio

async def procesar_chunk(chunk, texto_base, llm):
    # Llamada asíncrona al LLM
    respuesta = await llm.ainvoke(...)
    return f"[Contexto: {respuesta.content}] {chunk.page_content}"

async def contextualizar_todos(chunks, texto_base, llm):
    tareas = [procesar_chunk(c, texto_base, llm) for c in chunks]
    # Ejecuta todas las llamadas a la API de OpenAI en paralelo
    resultados = await asyncio.gather(*tareas) 
    return resultados
```

**Impacto en la recuperación actual (`retrieval.py`):**
- **Nulo / Transparente.** No tienes que cambiar absolutamente NADA en tus funciones `buscar_hibrido`, `buscar_similitud` o BM25. Tu código actual funcionará igual, pero entregará resultados muchísimo más precisos porque el texto indexado ya incluye el contexto.

---

## 2. Parent-Child Chunking (Auto-merging Retriever)

**Concepto:**
Buscas precisión en la recuperación (fragmentos cortos) pero amplio contexto para que el LLM responda (fragmentos largos).

**Implementación:**
1. Divides el documento en "Padres" (ej. 1,000 tokens).
2. Divides cada "Padre" en "Hijos" (ej. 200 tokens).
3. Guardas los Hijos en ChromaDB con un metadato: `{"parent_id": "ID_DEL_PADRE"}`.
4. Guardas los Padres en una base de datos clave-valor simple (ej. un diccionario o InMemoryStore).

**Impacto en la recuperación actual (`retrieval.py`):**
- **Requiere modificaciones.** Tu clase `MotorBusqueda` tendría que cambiar. Cuando ejecutas `self.vectorstore.similarity_search()`, ChromaDB te devolvería los "Hijos". Antes de mandar esos documentos al LLM para la respuesta final, tendrías que leer el `parent_id` de cada Hijo y buscar el texto completo del Padre en tu diccionario para pasárselo al LLM. (LangChain tiene una clase preconstruida para esto llamada `ParentDocumentRetriever`).

---

## 3. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

**Concepto:**
Resuelve el problema de las consultas globales que requieren entender todo el documento (ej. "Resume las obligaciones principales de todo el manual"). Crea un árbol de resúmenes.

**Implementación:**
1. Divides en chunks normales (Hojas).
2. Agrupas semánticamente los chunks similares y usas un LLM para resumir cada grupo (Nodos Nivel 1).
3. Agrupas los resúmenes y los resumes (Nodos Nivel 2)... hasta llegar a la Raíz.
4. Metes *absolutamente todo* (Hojas y Resúmenes) a ChromaDB.

**Impacto en la recuperación actual (`retrieval.py`):**
- **Requiere configuración de umbrales.** Tu código `buscar_hibrido` funcionaría de caja porque los resúmenes son solo texto. Sin embargo, podrías recuperar una mezcla de "resúmenes" y "fragmentos originales" al mismo tiempo. A veces esto es bueno, pero requiere afinar (tunear) el algoritmo para evitar duplicidad de información en el prompt final. Además, tu ingesta (chunking) se vuelve mucho más compleja.

---

## Conclusión y Recomendación

Dado el estado actual de tu proyecto:
1. **Quédate con Contextual Retrieval.** Es la estrategia más robusta y la que **MENOS** va a romper o afectar tu código actual de recuperación (`retrieval.py`).
2. El único cambio que requiere hacer es reescribir tu función `ContextualizadorLLM.procesar` en `chunking.py` para que use `asyncio`. Eso solucionará el problema de las 5 horas y te permitirá iterar y poblar tu base de datos en cuestión de minutos.
