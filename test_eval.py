from pathlib import Path
import os

from src.nlp_core.vectorizacion import MotorVectorizacion
from src.nlp_core.retrieval import MotorBusqueda
from src.nlp_core.evals.evaluador import EvaluadorRAG
from src.nlp_core.pipeline import PipelineRecuperacion

# 1. Configuración de Rutas
base_dir = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()
chroma_path = base_dir / "data" / "02_vectorstore" / "chroma_db"
gt_path = base_dir / "data" / "evaluacion_dataset.json"

# 2. Inicializar Componentes de Búsqueda
motor_vectorizacion = MotorVectorizacion(persist_directory=str(chroma_path))
motor_busqueda = MotorBusqueda(motor_vectorizacion)

# Configurar Pipeline Híbrido Simple
pipeline = PipelineRecuperacion(motor_busqueda)
pipeline.configurar(
    usar_expansion=False,
    usar_busqueda_hibrida=True,
    usar_cross_encoder=False
)
def buscar_hibrido_pipeline(query: str, k: int = 10):
    return pipeline.buscar(query, k=k)

# 3. Inicializar Evaluador
evaluador = EvaluadorRAG(ground_truth_path=gt_path)

print("\n--- PRUEBA DE MODOS DE EVALUACION (1 CONSULTA) ---\n")

print("1. Modo: EXACT MATCH (Linea Base)")
evaluador.evaluar_retrieval(
    funcion_busqueda=buscar_hibrido_pipeline,
    estrategia_nombre="Hibrido_Prueba",
    modo_evaluacion="exact_match",
    limite_consultas=1,
    k=3
)

print("2. Modo: HUMAN (Auditoría)")
evaluador.evaluar_retrieval(
    funcion_busqueda=buscar_hibrido_pipeline,
    estrategia_nombre="Hibrido_Prueba",
    modo_evaluacion="human",
    limite_consultas=1,
    k=3
)

print("3. Modo: LLM JUDGE (Evaluacion Semantica)")
evaluador.evaluar_retrieval(
    funcion_busqueda=buscar_hibrido_pipeline,
    estrategia_nombre="Hibrido_Prueba",
    modo_evaluacion="llm_judge",
    limite_consultas=1,
    k=3
)
