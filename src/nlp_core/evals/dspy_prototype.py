"""
dspy_prototype.py

Este es un prototipo conceptual de cómo integrar DSPy (Stanford) 
en el proyecto "proyecto_disf_npl".
La idea es reemplazar el "Prompt Engineering Manual" por "Programación LLM",
donde DSPy optimiza automáticamente los prompts basándose en una métrica (ej. Faithfulness).
"""

import os
import dspy
import pickle
from pathlib import Path
from pydantic import BaseModel, Field

# 1. Configurar el Modelo de Lenguaje (DSPy v3 soporta OpenAI vía LiteLLM)
# Asegúrate de tener OPENAI_API_KEY en tus variables de entorno (.env)
import os
from dotenv import load_dotenv
load_dotenv()

# Usamos la sintaxis dspy.LM introducida en DSPy v3
llm_model = dspy.LM('openai/gpt-4o-mini', max_tokens=1000)
dspy.settings.configure(lm=llm_model)

# 2. Cargar el Dataset (Caché Semántico)
# Usaremos las consultas que funcionaron bien en el pasado como nuestro "Training/Validation Set"
def cargar_dataset_dspy():
    cache_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "03_output" / "semantic_cache.pkl"
    dataset = []
    
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            datos = pickle.load(f)
            for item in datos:
                # Cada Example necesita los inputs y outputs esperados
                # Para QA, el input es (contexto, pregunta) y el output esperado es la (respuesta)
                
                # Unir los chunks en un solo texto de contexto
                contexto_str = "\n".join([c["content"] for c in item.get("chunks", [])])
                pregunta = item.get("query", "")
                respuesta = item.get("respuesta", "")
                
                if contexto_str and pregunta and respuesta:
                    ejemplo = dspy.Example(
                        context=contexto_str, 
                        question=pregunta, 
                        answer=respuesta
                    ).with_inputs('context', 'question')
                    
                    dataset.append(ejemplo)
                    
    return dataset

# 3. Definir la "Signature" (La firma de la función, en lugar de un prompt largo)
class GeneracionBasadaEnNormativa(dspy.Signature):
    """Responde preguntas basadas estrictamente en la normativa (contexto) proporcionada."""
    context = dspy.InputField(desc="Fragmentos de normativa y regulaciones financieras recuperadas.")
    question = dspy.InputField(desc="Pregunta del usuario sobre la normativa.")
    answer = dspy.OutputField(desc="Respuesta clara, estructurada y basada ÚNICAMENTE en el contexto.")

# 4. Definir el Módulo (El "Programa")
class RAGNormativo(dspy.Module):
    def __init__(self):
        super().__init__()
        # dspy.ChainOfThought le dice al LLM que "piense paso a paso" antes de generar la salida final
        self.generar_respuesta = dspy.ChainOfThought(GeneracionBasadaEnNormativa)
        
    def forward(self, context, question):
        # Aquí se podría incluir el código de ChromaDB, pero como ya tenemos el contexto,
        # simplemente llamamos al módulo generador.
        prediccion = self.generar_respuesta(context=context, question=question)
        return dspy.Prediction(answer=prediccion.answer)

# 5. Métrica de Evaluación (Automática)
# En lugar de usar RAGAS manualmente, definimos una métrica que DSPy usará para optimizar.
# En este caso, usamos un modelo como "Juez" para validar si la respuesta no tiene alucinaciones.
class ValidacionFaithfulness(dspy.Signature):
    """Evalúa si la respuesta generada está completamente sustentada por el contexto."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.InputField()
    is_faithful = dspy.OutputField(desc="Responde 'True' si la respuesta se basa 100% en el contexto, o 'False' si hay alucinación.")

def metrica_faithfulness(example, pred, trace=None):
    # Usar un LLM juez para evaluar
    evaluador = dspy.Predict(ValidacionFaithfulness)
    resultado = evaluador(context=example.context, question=example.question, answer=pred.answer)
    return 1.0 if "True" in str(resultado.is_faithful) else 0.0

# 6. Optimizador (Teleprompter)
# Esto "Compila" nuestro módulo RAGNormativo ajustando el prompt interno para maximizar la métrica
def optimizar_pipeline():
    from dspy.teleprompt import BootstrapFewShot
    
    print("🚀 Cargando dataset desde el Caché Semántico...")
    dataset = cargar_dataset_dspy()
    
    if len(dataset) < 5:
        print("⚠️ No hay suficientes ejemplos en el caché para optimizar (se recomiendan al menos 5).")
        return None
        
    print(f"📊 Dataset cargado con {len(dataset)} ejemplos.")
    
    # Configuramos el optimizador para usar few-shot learning
    # BootstrapFewShot simula llamadas y guarda los mejores ejemplos que maximizan la métrica
    optimizador = BootstrapFewShot(
        metric=metrica_faithfulness, 
        max_bootstrapped_demos=4, 
        max_labeled_demos=16
    )
    
    rag_base = RAGNormativo()
    
    print("🧠 Compilando y optimizando el modelo RAG...")
    # El proceso de compilación tomará nuestro pipeline base y creará uno optimizado
    rag_optimizado = optimizador.compile(rag_base, trainset=dataset)
    
    print("✅ ¡Pipeline compilado exitosamente!")
    # Guardamos el modelo optimizado (esto guarda los prompts óptimos que DSPy descubrió)
    output_path = Path(__file__).resolve().parent / "rag_normativo_optimizado.json"
    rag_optimizado.save(str(output_path))
    
    print(f"\n💾 El modelo optimizado (con sus few-shot examples) se ha guardado en:\n{output_path}")
    print("\n🔍 Para ver exactamente cómo se ve el prompt final bajo el capó, puedes revisar:")
    print("El historial del LLM con: dspy.settings.lm.inspect_history(n=1)")
    
    return rag_optimizado

if __name__ == "__main__":
    print("--- INICIANDO PROTOTIPO DSPY ---")
    print("Este script demuestra cómo DSPy usaría el historial (Caché Semántico)")
    print("para auto-descubrir los mejores prompts sin intervención humana.\n")
    optimizado = optimizar_pipeline()
    
    if optimizado:
        print("\n--- ÚLTIMO PROMPT ENVIADO A OPENAI POR DSPY ---")
        try:
            llm_model.inspect_history(n=1)
        except Exception:
            pass
