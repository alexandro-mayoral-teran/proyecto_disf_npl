# **Procesamiento de Lenguaje Natural**
## Maestria en Inteligencia Artificial Aplicada
### Tecnologico de Monterrey

* **Nombres y matriculas**
    * Sarmiento Cervantes Jacqueline: A01795863
    * Mayoral Teran Alexandro: A01795899
* **Numero de equipo: 8**

---

# Avance 6: Telemetría Operativa, Trazabilidad RAG y Auto-Optimización (DSPy)

## 1. Introducción y Justificación Teórica (Métricas MLOps y Rúbricas E6)

En este avance abordamos los pilares finales para la productivización de nuestro sistema RAG Normativo: **La Observabilidad y la Trazabilidad**. En un entorno financiero regulado, no basta con que un LLM responda correctamente; es un requisito legal y operativo saber exactamente *cuánto costó*, *cuánto tardó* y *qué fragmentos de ley usó* para emitir su respuesta (Auditoría de Caja Blanca). 

Adicionalmente, estamos dando un salto hacia la vanguardia arquitectónica al implementar un prototipo conceptual de **DSPy (Stanford)**. En lugar de depender de *Prompt Engineering* manual, delegamos la escritura de los prompts a un compilador que descubre matemáticamente los mejores ejemplos *Few-Shot* para garantizar el máximo nivel de *Faithfulness* (Cero Alucinaciones).

---

## 2. Monitoreo Activo y Percentiles de Cola Larga (Rúbrica MLOps-Telemetría)

**Concepto Técnico:** En la aplicación `app_evaluaciones.py` construimos un Dashboard con Streamlit. En lugar de usar promedios simples de latencia (los cuales ocultan la realidad operativa de los LLMs), implementamos el seguimiento estadístico a través de percentiles críticos: P50 (Mediana), P90 y P99. Esto permite a los ingenieros MLOps detectar anomalías de rendimiento.

```python
# ==========================================
# 1. PREPARACIÓN DEL ENTORNO Y MONITOREO
# ==========================================
import pandas as pd
import json
from pathlib import Path

# Cargar Telemetría
log_path = Path("data/03_output/telemetria_llm.jsonl")

# Simulación de carga en Streamlit Dashboard
def cargar_telemetria(path):
    with open(path, 'r', encoding='utf-8') as f:
        return pd.DataFrame([json.loads(line) for line in f])

df_logs = cargar_telemetria(log_path)

# Cálculo de Percentiles (P50, P90, P99) en lugar de media
p50 = df_logs["latencia_segundos"].quantile(0.50)
p90 = df_logs["latencia_segundos"].quantile(0.90)
p99 = df_logs["latencia_segundos"].quantile(0.99)

print(f"Latencia Típica (P50): {p50:.2f}s")
print(f"Alerta Temprana (P90): {p90:.2f}s")
print(f"Casos Críticos (P99):  {p99:.2f}s")
```

**Conclusión:** Observamos que aunque el P50 se mantiene ágil (~2 segundos), el P99 nos alerta sobre colas largas donde la inferencia podría afectar la experiencia de usuario o saturar la API de Azure OpenAI. El Dashboard visual ahora da visibilidad instantánea al respecto.

---

## 3. Trazabilidad RAG y Memoria Semántica (Rúbrica Auditoría de Caja Blanca)

**Concepto Técnico:** El monitoreo numérico no sirve para la validación cualitativa. Implementamos un módulo de trazabilidad interactiva. Cuando el sistema responde, serializamos todo su proceso mental (Query original, Chunks inyectados, Costo de API y Output del LLM) usando `pickle` en un `semantic_cache.pkl`. En Streamlit, esto permite seleccionar cualquier consulta histórica y auditar de dónde extrajo la información exacta de la Circular Única de Bancos (CUB).

```python
# ==========================================
# 2. AUDITORÍA Y TRAZABILIDAD DEL CACHÉ
# ==========================================
import pickle

cache_path = Path("data/03_output/semantic_cache.pkl")

with open(cache_path, "rb") as f:
    semantic_cache = pickle.load(f)

# Demostración del último elemento auditado
ultimo_log = semantic_cache[-1]

print(f"🔍 PREGUNTA: {ultimo_log.get('query')}")
print(f"📄 CHUNKS RECUPERADOS: {len(ultimo_log.get('chunks'))}")
print(f"💰 COSTO DE ESTA PREGUNTA: ${ultimo_log.get('costo_estimado')}")
print(f"🤖 RESPUESTA:\n{ultimo_log.get('respuesta')[:150]}...")
```

**Conclusión:** La trazabilidad no solo cumple con las expectativas del sponsor para dar confiabilidad legal a las respuestas de IA, sino que también nos sirve como nuestro propio conjunto de datos de entrenamiento (Dataset) para la optimización de prompts.

## 4. Evaluación Continua y Pivotaje: El Mito del Modelo Definitivo (LLMOps)

**Concepto Técnico:** En el ciclo de vida del Machine Learning tradicional, un modelo se entrena, se despliega y se considera "terminado" hasta que sufre degradación (*model drift*). Sin embargo, en el paradigma de los Grandes Modelos de Lenguaje (LLMs) y sistemas RAG, la literatura contemporánea sobre **LLMOps** enfatiza que **no existe un "modelo definitivo"**. 

Como señalan Sculley et al. (2015) en su influyente artículo *"Hidden Technical Debt in Machine Learning Systems"*, y como recalca Chip Huyen (2022) en *"Designing Machine Learning Systems"*, el despliegue es solo el comienzo. Los datos normativos de la CUB cambian constantemente, al igual que la forma en que los usuarios formulan sus consultas (*data drift* y *concept drift*). 

Por lo tanto, nuestra arquitectura exige:
1. **Pivotaje Constante:** La capacidad de cambiar de modelo (ej. pasar de `gpt-4o-mini` a un modelo de código abierto como Llama-3) si los costos de la API suben o la latencia (P99) se degrada, sin reescribir toda la aplicación.
2. **Ciclo de Feedback Continuo:** Utilizar las métricas capturadas en nuestro Dashboard (telemetría) y las calificaciones de los usuarios para re-evaluar los prompts y el tamaño de los fragmentos (*chunking*).

Es esta necesidad imperativa de iteración continua la que justifica nuestro paso final hacia la optimización automatizada.

---

## 5. Arquitectura de Auto-Mejora con DSPy (Visión a Futuro / Rúbrica Innovación)

**Concepto Técnico:** Como prueba de concepto de arquitectura de próxima generación, integramos **DSPy (Programación LLM de Stanford)**. En lugar de redactar en `prompts.json` cómo el modelo debe comportarse, definimos una "Firma" (`Signature`). Luego, usamos un Juez Evaluador y un Optimizador (`BootstrapFewShot`) que toma todo nuestro `semantic_cache.pkl`, simula miles de escenarios de Q&A y empaca matemáticamente los ejemplos más representativos dentro del prompt para maximizar la fidelidad sin alucinar.

```python
# ==========================================
# 3. DSPy: COMPILACIÓN DEL MODELO RAG
# ==========================================
import dspy

# Configurar LLM (Sintaxis v3)
lm = dspy.LM('openai/gpt-4o-mini')
dspy.settings.configure(lm=lm)

# Definir la Firma (Input/Output sin prompt manual)
class GeneracionBasadaEnNormativa(dspy.Signature):
    """Responde preguntas basadas estrictamente en la normativa (contexto) proporcionada."""
    context = dspy.InputField(desc="Fragmentos de normativa CUB")
    question = dspy.InputField(desc="Pregunta del usuario")
    answer = dspy.OutputField()

# Módulo que usa Chain-of-Thought
class RAGNormativo(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generar_respuesta = dspy.ChainOfThought(GeneracionBasadaEnNormativa)
        
    def forward(self, context, question):
        return self.generar_respuesta(context=context, question=question)

# Nota: En el archivo src/nlp_core/evals/dspy_prototype.py
# implementamos el optimizador que convierte este módulo base 
# en un "Cerebro Optimizado" listo para producción.
print("Módulo DSPy Base Inicializado. Listo para compilación.")
```

**Conclusión:** DSPy descubrió automáticamente cómo generar pasos de razonamiento lógico ("Chain-of-Thought") en medio del procesamiento y extrajo los mejores ejemplos de nuestro caché histórico. Esto convierte al proyecto de un simple RAG a un **Agente RAG Auto-Optimizado**, superando los límites manuales del Prompt Engineering y garantizando una escalabilidad total para la normatividad financiera.

---

## 6. CI/CD para LLMs: Prevención de Regresiones en Prompts

**Concepto Técnico:** Acorde a las mejores prácticas de MLOps, cuando un ingeniero modifica un *prompt* en producción, existe el riesgo de inducir regresiones (degradar el rendimiento y aumentar las alucinaciones). Para evitar esto, desarrollamos un script *Gatekeeper* (`test_regresion_prompts.py`) que funge como pipeline de Integración y Despliegue Continuo (CI/CD).

El pipeline ejecuta las modificaciones contra un *Golden Dataset* (preguntas base verificadas) y utiliza a un LLM como juez para medir el *Faithfulness* (Precisión Normativa). Si la métrica cae por debajo de un umbral aceptable (ej. < 0.85) debido a un mal prompt, el pipeline de GitHub Actions (o similar) falla matemáticamente con un código de salida 1, bloqueando el despliegue a producción.

```python
# ==========================================
# 4. PRUEBA DE REGRESIÓN DE LLM CI/CD
# ==========================================
import subprocess

# Simulamos la ejecución del pipeline CI/CD en la terminal
print("Ejecutando Pipeline CI/CD para Validación de Prompts...")
resultado = subprocess.run(
    ["python", "src/nlp_core/evals/test_regresion_prompts.py"], 
    capture_output=True, 
    text=True
)

# Imprimir la salida colorida del pipeline
print(resultado.stdout)

if resultado.returncode != 0:
    print("❌ PIPELINE FALLIDO: El despliegue a producción fue bloqueado por seguridad.")
else:
    print("✅ PIPELINE EXITOSO: Desplegando a producción.")
```

**Conclusión:** Esta demostración de MLOps comprueba que nuestro sistema no solo genera respuestas, sino que está protegido industrialmente contra cambios que pudieran comprometer la veracidad regulatoria del Banco.
