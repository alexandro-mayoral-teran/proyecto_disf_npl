import os
import sys
import json
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.nlp_core.config_llm import get_llm_client, get_llm_model_name

def calcular_similitud_respuestas(respuestas: list[str]) -> float:
    """
    Evalúa qué tan consistentes son las N respuestas generadas para la misma pregunta.
    Utiliza el LLM Juez para medir si las respuestas son semánticamente equivalentes.
    Retorna un score de 0.0 (totalmente distintas) a 1.0 (idénticas en significado).
    """
    client = get_llm_client("qa")
    
    if len(respuestas) < 2: return 1.0
    
    # Comparamos la respuesta 0 contra las demás
    base = respuestas[0]
    acuerdos = 0
    comparaciones = 0
    
    for resp_alt in respuestas[1:]:
        prompt = f"""
        Eres un juez de consistencia semántica.
        Evalúa si estas dos respuestas a una misma consulta normativa contienen exactamente los mismos hechos, aunque estén redactadas distinto.
        
        Respuesta 1: {base}
        Respuesta 2: {resp_alt}
        
        Responde ÚNICAMENTE con 'SIMILAR' o 'DISTINTA'.
        """
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            veredicto = res.choices[0].message.content.strip().upper()
            if "SIMILAR" in veredicto:
                acuerdos += 1
        except Exception as e:
            pass
        comparaciones += 1
        
    return acuerdos / comparaciones if comparaciones > 0 else 0.0

def correr_evaluacion_consistencia(query_prueba: str, n_runs: int = 3, temperatura: float = 0.7):
    """
    Mide la calibración y 'Self-Consistency' de un modelo. 
    (Rúbrica ENS-E).
    """
    print(f"Iniciando Prueba de Consistencia (Temperatura={temperatura}, Runs={n_runs})")
    client = get_llm_client("qa")
    modelo = get_llm_model_name("qa")
    
    respuestas = []
    
    for i in range(n_runs):
        print(f" - Corrida {i+1}/{n_runs}...")
        try:
            res = client.chat.completions.create(
                model=modelo,
                messages=[{"role": "user", "content": query_prueba}],
                temperature=temperatura
            )
            respuestas.append(res.choices[0].message.content)
        except Exception as e:
            print(f"Error en corrida {i+1}: {e}")
            
    # Calcular Consistencia (Paraphrase Invariance)
    consistency_score = calcular_similitud_respuestas(respuestas)
    
    resumen = {
        "modelo": modelo,
        "query": query_prueba,
        "temperatura": temperatura,
        "n_runs": n_runs,
        "consistency_score": float(consistency_score),
        "interpretacion": "Altamente Consistente" if consistency_score > 0.8 else "Miscalibrado / Alucinación Probable"
    }
    
    print("-" * 50)
    print(f"Modelo: {modelo}")
    print(f"Score de Consistencia (Self-Consistency): {consistency_score*100:.2f}%")
    print(f"Diagnostico: {resumen['interpretacion']}")
    print("-" * 50)
    
    return resumen

if __name__ == "__main__":
    # Prueba de humo
    query = "¿Cuales son las sanciones por no enviar el reporte regulatorio de tarjetas de crédito a tiempo?"
    correr_evaluacion_consistencia(query)
