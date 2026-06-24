# Guía Rápida: Cómo Interpretar los Resultados de Evaluación

Esta guía explica cómo leer y analizar los CSVs que se generan en la carpeta `data/03_output/evaluaciones/` tras correr el script de evaluación (`evaluador_integral.py`).

## 1. Fase 1: Arena de Retrieval (`ARENA_RESULTADOS_...csv`)
Este archivo compara el rendimiento puro de tus distintas estrategias de búsqueda (BoW, Embeddings, Híbrido, etc.).

- **Recall@10:** Porcentaje de veces que el documento correcto (definido en el Ground Truth) apareció entre los primeros 10 resultados recuperados. Ejemplo: Un 85% significa que el sistema es muy robusto para encontrar la "aguja en el pajar".
- **MAP@10 / NDCG@10:** Miden la *calidad del orden*. No basta con encontrar el documento; es vital ponerlo en la posición #1 o #2. Valores más cercanos a 1.0 significan que la respuesta correcta casi siempre es el primer resultado que se le envía al LLM.
- **CI_95 (Intervalos Bootstrap):** Muestran el rango de confianza estadístico. Un Recall de `85% [CI: 81% - 88%]` significa que puedes garantizarle a los *sponsors* con 95% de seguridad que el sistema no bajará del 81% de efectividad en el mundo real.
- **Latencia_Promedio_Segundos:** Cuánto tardó la fase de búsqueda (sin contar al LLM). Es vital para la toma de decisiones, ya que una estrategia súper precisa como *Híbrido + CrossEncoder* podría resultar ser muy lenta para la experiencia de usuario.

## 2. Fase 2: Prueba Ciega o Contaminación (`contaminacion_ciega_...csv`)
Este reporte te dice si tu LLM "hizo trampa", es decir, si pudo contestar la pregunta tirando de su propia memoria sin necesidad del RAG.

- **Precision_Ciega_Porcentaje:** 
  - **Si es Alto (ej. > 40%):** El modelo ya conocía esa normativa de antemano.
  - **Si es Bajo (0% - 10%):** ¡Es un resultado excelente! Significa que el modelo depende 100% de la información recuperada por tu RAG para responder correctamente. Valida que todo el esfuerzo de tu arquitectura de búsqueda realmente aporta valor crítico.

## 3. Fase 3: Análisis de Errores Desagregado (`analisis_errores_desagregados_...csv`)
En un RAG complejo, cuando una extracción falla, es difícil saber por qué. Este archivo clasifica cada error en 3 categorías mutuamente excluyentes para saber *de quién es la culpa*:

- **Fallo A (Culpa del Buscador / Retrieval):** El buscador nunca logró recuperar el fragmento normativo correcto para pasárselo al LLM. 
  - *Solución:* Mejorar la estrategia de Embeddings, agregar expansión de queries (HyDE) o ajustar el tamaño de los chunks.
- **Fallo B (Culpa del LLM / Alucinación):** El buscador hizo bien su trabajo y le entregó el texto perfecto, pero el LLM se confundió, alucinó o razonó mal la respuesta. 
  - *Solución:* Cambiar a un modelo más inteligente (ej. LLaMA 70B o GPT-4o) o mejorar la claridad de las reglas en el Prompt del sistema.
- **Fallo C (Culpa de Formato / Parser JSON):** El buscador le dio la info correcta y el LLM extrajo bien los datos, pero falló al armar el esquema JSON (olvidó unas comillas, usó un string en vez de int, etc.) rompiendo la validación de Pydantic. 
  - *Solución:* Bajar la temperatura a `0.0`, usar LLMs afinados para *Structured Outputs* o utilizar la validación matemática nativa (`client.beta.chat.completions.parse`) para forzar la estructura.

## 4. Evaluaciones de Robustez y Ensambles (Avance 5)
Para cumplir con los estándares de MLOps en el Avance 5, se implementaron evaluaciones científicas adicionales a través de scripts especializados en `src/lab/`:

- **Diversidad del Ensamble (Correlación de Pearson):** Mide empíricamente si combinar dos modelos de búsqueda (ej. BM25 y Embeddings) vale la pena. Una correlación baja (<0.50) significa que los modelos cometen errores en documentos distintos, por lo que fusionarlos (vía Reciprocal Rank Fusion) es estadísticamente superior a usarlos por separado.
- **Calibración y Self-Consistency (Proxy ECE):** Evaluamos qué tan seguro está el LLM de sus propias respuestas corriendo la misma consulta múltiples veces (`temperature > 0`). Una varianza alta significa baja consistencia (el modelo "adivina" y alucina). Un modelo bien calibrado debe producir respuestas consistentes.
- **Frontera de Pareto (TCO vs Calidad):** Analiza gráficamente la relación entre Latencia/Costo ($) en el eje X y Precisión (NDCG@10) en el eje Y. Te ayuda a justificar arquitecturas como el **Model Cascading** (usar un modelo local gratuito primero, y delegar a la nube solo si la confianza es baja) demostrando que mantiene un NDCG alto recortando el costo drásticamente.

## Trazabilidad (Metadatos)
En todos los CSVs verás columnas inyectadas como `LLM_Modelo_QA`, `Es_Local` o `Git_Hash`. Utiliza estas variables como dimensiones cuando cruces los datos en Excel/Python para dibujar tu **Frontera de Pareto** (graficando `NDCG@10` vs. `Costo` vs. `Latencia`).

## 5. Métricas Satelitales (Estilo RAGAS / Reference-Free)
Además de las pruebas contra el Ground Truth humano, el orquestador (`src/nlp_core/evals/evaluador.py`) expone métricas Reference-Free donde un Juez LLM evalúa la calidad sin conocer la respuesta perfecta:
- **Faithfulness (Fidelidad):** Evalúa si la respuesta generada se deriva estrictamente del contexto recuperado. Si es baja (< 0.8), significa que el modelo está alucinando información externa.
- **Answer Relevance (Relevancia):** Mide si la respuesta atiende directamente la pregunta original o si la está evadiendo con respuestas vagas.
- **Context Relevancy (Utilidad del Contexto):** Mide qué porcentaje de las oraciones recuperadas son realmente útiles. Un score bajo indica "ruido" excesivo en la base vectorial, lo que encarece los costos de inferencia.
