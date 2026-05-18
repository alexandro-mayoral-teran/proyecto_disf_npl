# Avance 3: Mapeo de Conceptos ML a Generative AI / RAG

Dado que las instrucciones de la maestría están orientadas a Machine Learning predictivo clásico (clasificación/regresión), a continuación se presenta la "traducción" académica de esos conceptos a nuestro proyecto de Procesamiento de Lenguaje Natural (NLP) e Inteligencia Artificial Generativa.

---

### 1. ¿Qué algoritmo se puede utilizar como baseline para resolver el problema?
**En ML Tradicional:** Regresión Logística, Random Forest simple.
**En Nuestro Proyecto (RAG/LLM):** 
Nuestro "Baseline" (Línea Base) es el enfoque de **Full Context Zero-Shot Prompting**. Es decir, pasar el documento normativo completo directamente a la ventana de contexto de un LLM base (como `gpt-4o-mini`) sin usar bases de datos vectoriales ni búsqueda híbrida. 
Este baseline servirá para evaluar si la complejidad de construir el RAG (nuestro modelo avanzado) realmente justifica la inversión frente a "simplemente pasarle el texto al LLM".

### 2. ¿Se puede determinar la importancia de las características (Feature Importance)?
**En ML Tradicional:** Ponderación de variables en un árbol de decisión.
**En Nuestro Proyecto (RAG/LLM):** 
La importancia de las "características" se traduce en la **relevancia del contexto recuperado** (Context Relevance). En el próximo Notebook, evaluaremos si recuperar fragmentos basados en coincidencias exactas (BM25 - léxico) aporta más valor que la comprensión abstracta (ChromaDB - semántico). El peso que el algoritmo *Reciprocal Rank Fusion (RRF)* le asigne a cada método es el equivalente a la "importancia" de la característica.

### 3. ¿El modelo está sub/sobreajustando los datos (Underfitting/Overfitting)?
**En ML Tradicional:** Desempeño perfecto en entrenamiento pero malo en validación.
**En Nuestro Proyecto (RAG/LLM):** 
*   **Sobreajuste (Overfitting) = Alucinaciones:** El LLM "memoriza" conceptos financieros generales y los inventa en el esquema final, asumiendo que el Banco de México los pide cuando en realidad el texto no los menciona.
*   **Subajuste (Underfitting) = Omisiones:** El sistema RAG falla en recuperar el fragmento correcto, provocando que el LLM devuelva un JSON incompleto (falta de exhaustividad).

### 4. ¿Cuál es la métrica adecuada para este problema de negocio?
**En ML Tradicional:** Accuracy, MSE, R2.
**En Nuestro Proyecto (RAG/LLM):** 
Las métricas se dividen en dos dimensiones críticas para la DISF:
1.  **Métricas de Calidad de Extracción (vs Golden Dataset):**
    *   *Recall (Exhaustividad):* % de variables y fórmulas correctamente extraídas frente al Excel manual.
    *   *Precision:* % de campos extraídos que realmente existen en el documento (penaliza alucinaciones).
2.  **Métricas Operativas (Telemetría):**
    *   *Latencia (segundos):* Tiempo de respuesta.
    *   *Eficiencia de Tokens:* Costo en USD por consulta.

### 5. ¿Cuál debería ser el desempeño mínimo a obtener?
El Baseline (*Full Context*) nos dará el "piso". El modelo avanzado (*RAG Híbrido*) debe lograr, como mínimo:
*   Un **Recall superior al 85%** en la extracción de variables financieras complejas.
*   Una reducción de **al menos 50% en el consumo de Input Tokens** frente al Baseline.
*   Cero (0%) alucinaciones en la generación de fórmulas de cálculo.
