import json
import pandas as pd
from pathlib import Path
from src.nlp_core.evals.evaluador import EvaluadorRAG
from dotenv import load_dotenv

load_dotenv()

class MockDoc:
    def __init__(self, content):
        self.page_content = content

def mock_search(query, k):
    return [MockDoc("Este documento falso contiene A los créditos que sean otorgados a personas físicas para test.")]

base_dir = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()
gt_path = base_dir / "data" / "evaluacion_dataset.json"

evaluador = EvaluadorRAG(ground_truth_path=gt_path)

print("1. Modo: EXACT MATCH")
evaluador.evaluar_retrieval(mock_search, "Mock", modo_evaluacion="exact_match", limite_consultas=1, k=1)

print("2. Modo: HUMAN")
evaluador.evaluar_retrieval(mock_search, "Mock", modo_evaluacion="human", limite_consultas=1, k=1)

print("3. Modo: LLM JUDGE")
evaluador.evaluar_retrieval(mock_search, "Mock", modo_evaluacion="llm_judge", limite_consultas=1, k=1)
