# Justificación Técnica del Modelo Local: LLaMA 3.1 (8B)

En la arquitectura de nuestro pipeline RAG Híbrido, hemos seleccionado el modelo **LLaMA 3.1 de 8 mil millones de parámetros (`llama3.1:8b`)** de Meta para operar localmente a través de Ollama. 

A continuación, se documenta la justificación formal para esta decisión de diseño arquitectónico, evaluando sus ventajas, desventajas y comparativa con el mercado, con el fin de tener certidumbre técnica frente a los *sponsors* del proyecto.

---

## 1. ¿Por qué LLaMA 3.1 8B es la mejor opción actual? (Frontera de Pareto)

La elección de un modelo local para tareas NLP corporativas (como normativas de Banxico/DISF) se reduce a un problema de optimización entre **Hardware Requerido vs. Razonamiento Lógico**. 

LLaMA 3.1 8B se sitúa exactamente en el "Sweet Spot" de esta frontera:

### Ventajas Principales
1. **Ventana de Contexto Masiva (128k Tokens):** A diferencia de modelos anteriores que colapsaban a los 4k u 8k tokens, LLaMA 3.1 puede ingerir hasta 128,000 tokens. Esto es **crítico para RAG**, ya que nos permite inyectar decenas de fragmentos normativos recuperados sin truncar la memoria del modelo.
2. **Hardware de Consumo:** Al estar cuantizado a 4-bits o 8-bits mediante Ollama, este modelo requiere entre **4.5 GB y 8 GB de RAM/VRAM** para funcionar. Esto significa que puede correr en la laptop de un analista o en un servidor On-Premise sin necesidad de comprar costosos clusters de GPUs (A100/H100).
3. **Capacidad Multilingüe Mejorada:** A diferencia de LLaMA 2 o LLaMA 3 original, la versión 3.1 fue entrenada explícitamente con un fuerte enfoque multilingüe, logrando un manejo del **español normativo/legal** excepcionalmente fluido y preciso.
4. **Instrucciones Estrictas (Tool Use & JSON):** La versión *Instruct* fue finamente ajustada para seguir reglas duras, lo cual es vital para tareas como la Expansión de Queries (HyDE) o el Análisis apegado a esquemas que requerimos.
5. **Licencia Permisiva:** Permite uso comercial (hasta 700 millones de usuarios activos), mitigando riesgos de licenciamiento si el proyecto escala a producción.

### Desventajas
1. **Extracción Estructurada Profunda:** Aunque es muy bueno respondiendo preguntas (QA), en la tarea de extracción de entidades complejas usando Pydantic (donde se exige un JSON anidado con Cero Errores sintácticos), aún se queda ligeramente atrás de modelos gigantes como GPT-4o. *(Por eso en nuestra arquitectura híbrida, la extracción compleja se delega a la nube)*.
2. **Lentitud en CPU:** Aunque cabe en la memoria RAM, sin una GPU discreta o Apple Silicon (M1/M2/M3), la velocidad de generación (tokens por segundo) puede ser frustrante para aplicaciones en tiempo real.

---

## 2. Comparativa contra Alternativas (Por qué los descartamos)

Para ser contundentes, evaluamos el mercado actual de modelos *Open-Weights* que caben en hardware estándar:

### ❌ Mistral v0.3 (7B) / Mixtral (8x7B)
- **Por qué no:** Mistral 7B fue el rey indiscutible hace un año, pero LLaMA 3.1 8B lo superó drásticamente en los *benchmarks* estándar (MMLU, HumanEval). Mixtral 8x7B tiene un rendimiento casi a la par de GPT-3.5, pero **requiere mucha más memoria (aprox. 24 GB de VRAM)** para correr eficientemente, destruyendo nuestro requerimiento de "hardware accesible".

### ❌ Gemma 2 (9B)
- **Por qué no:** Es el rival directo de Google. Tiene un rendimiento espectacular (incluso superior a Llama 3.1 en algunos tests de razonamiento). Sin embargo, su arquitectura requiere ligeramente más poder computacional y, lo más importante, su licencia de uso es **más restrictiva** para casos de uso corporativo/comercial en comparación con la de Meta. Además, LLaMA cuenta con una integración comunitaria nativa más robusta con herramientas como LangChain y ChromaDB.

### ❌ LLaMA 3.1 (70B) o Command R+
- **Por qué no:** Son los modelos de código abierto más inteligentes que existen, capaces de destronar a GPT-4 en varios rubros. Sin embargo, exigen hardware empresarial de ultra-alta gama (Múltiples GPUs de 80GB), lo cual anula el propósito de tener un modelo "Local ligero" enfocado en privacidad a bajo costo.

---

## 3. Conclusión Contundente

> [!IMPORTANT]
> **Veredicto:** LLaMA 3.1 8B no es solo una "buena" opción, es **la mejor opción estratégica actual** para nuestro caso de uso.

Nos provee un nivel de razonamiento casi a la par de modelos de la nube de hace apenas un año (GPT-3.5), con una ventana de contexto diseñada específicamente para RAG (128k), un excelente manejo del idioma español y la capacidad de correr de forma completamente privada en servidores estándar, sin arriesgar la exposición de los manuales regulatorios de la DISF. 

Al aislar `llama3.1:8b` para las tareas de **Generación Conversacional (QA)** y **Expansión de Consultas**, reservando la nube solo para la Extracción Estructurada pesada, hemos logrado un balance técnico perfecto.

> [!TIP]
> **Nota sobre Escalabilidad Futura:** Es importante recalcar que esta arquitectura y selección de modelos obedece a la **Fase de Prototipo (MVP)**, diseñada intencionalmente para demostrar valor bajo fuertes restricciones de hardware. La arquitectura de código está diseñada de manera agnóstica (`config_llm.py`), lo que significa que **si el proyecto se autoriza para producción y se asigna un presupuesto mayor** (ej. adquisición de servidores con GPUs robustas), es trivial sustituir este modelo por variantes superiores como **LLaMA 3.1 70B**, logrando capacidades de razonamiento nivel GPT-4 manteniéndose On-Premise.

---

## 4. Glosario de Conceptos Clave

Para mayor claridad técnica, definimos brevemente los términos utilizados en esta justificación:

- **Cuantización (Quantization):** Es una técnica de compresión. Los modelos de IA usualmente usan números muy precisos (16 o 32 bits) que ocupan muchísimo espacio. La cuantización los "redondea" a números más pequeños (ej. 4 u 8 bits). Esto reduce drásticamente la memoria RAM necesaria para correr el modelo y acelera su velocidad, perdiendo una fracción casi imperceptible de su "inteligencia". Ollama aplica esto automáticamente.
- **Ventana de Contexto:** Es la "memoria a corto plazo" del modelo. Mide cuántas palabras (tokens) puede leer y recordar al mismo tiempo antes de olvidar el inicio de la conversación. En RAG, una ventana grande es crucial para poder meterle muchos fragmentos de PDFs.
- **Open-Weights (Open-Source):** Modelos cuyos "pesos" matemáticos (el cerebro ya entrenado) son liberados públicamente, permitiendo a cualquiera descargarlos y ejecutarlos en sus propias máquinas sin pagar por uso de API ni depender de internet.
- **VRAM vs. RAM:** La VRAM es la memoria dedicada de las Tarjetas Gráficas (GPUs), que son muchísimo más rápidas para IA que los procesadores normales (CPUs) que usan RAM estándar. Usar modelos que quepan en la VRAM disponible es la clave del rendimiento local.
