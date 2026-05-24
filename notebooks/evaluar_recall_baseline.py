import os
import sys
import pandas as pd
from pathlib import Path

# Añadir el directorio raíz para importar src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.nlp_core.retrieval import MotorBusqueda
from src.nlp_core.evals.evaluador import EvaluadorRAG
from src.nlp_core.pipeline import PipelineRecuperacion
from langchain_core.documents import Document

def correr_la_arena(modo_evaluacion="exact_match", limite_consultas=None):
    # 1. Inicializar el Evaluador
    gt_path = project_root / "data" / "evaluacion_dataset.json"
    evaluador = EvaluadorRAG(ground_truth_path=gt_path)
    
    # 2. Inicializar el Motor de Búsqueda
    try:
        motor = MotorBusqueda()
        total_docs = motor.vectorstore._collection.count()
        if total_docs == 0:
            print("⚠️ ADVERTENCIA: Tu ChromaDB está vacío. Necesitas poblarlo primero.")
            return
    except Exception as e:
        print(f"❌ Error al inicializar MotorBusqueda: {e}")
        return

    # 3. Extraer documentos de ChromaDB para inicializar BM25
    print("\nExtrayendo corpus para los modelos lexicos...")
    raw_data = motor.vectorstore.get(include=["metadatas", "documents"])
    documentos_bm25 = [
        Document(page_content=txt, metadata=meta) 
        for txt, meta in zip(raw_data["documents"], raw_data["metadatas"])
    ]
    print(f"   Corpus cargado: {len(documentos_bm25)} fragmentos.")

    # 4. Definir las estrategias a evaluar usando el RAG Pipeline
    PIPELINES_A_EVALUAR = [
        {"nombre": "1_BoW", "base_retriever": "bow", "query_expansion": None, "post_processing": None},
        {"nombre": "2_TF-IDF", "base_retriever": "tfidf", "query_expansion": None, "post_processing": None},
        {"nombre": "3_BM25", "base_retriever": "bm25", "query_expansion": None, "post_processing": None},
        {"nombre": "4_Embeddings", "base_retriever": "embeddings", "query_expansion": None, "post_processing": None},
        {"nombre": "5_Hibrido_RRF", "base_retriever": "hibrido", "query_expansion": None, "post_processing": None},
        {"nombre": "6_Hibrido_CrossEncoder", "base_retriever": "hibrido", "query_expansion": None, "post_processing": "cross_encoder"},
        {"nombre": "7_Embeddings_CrossEncoder", "base_retriever": "embeddings", "query_expansion": None, "post_processing": "cross_encoder"},
        {"nombre": "8_MultiQuery_Embeddings_CrossEncoder", "base_retriever": "embeddings", "query_expansion": "multi_query", "post_processing": "cross_encoder"},
        {"nombre": "9_HyDE_Embeddings_CrossEncoder", "base_retriever": "embeddings", "query_expansion": "hyde", "post_processing": "cross_encoder"},
        {"nombre": "10_Ambos_Embeddings_CrossEncoder", "base_retriever": "embeddings", "query_expansion": "ambos", "post_processing": "cross_encoder"},
        {"nombre": "11_Ambos_Hibrido_CrossEncoder", "base_retriever": "hibrido", "query_expansion": "ambos", "post_processing": "cross_encoder"}
    ]

    resultados_arena = []

    print(f"\n--- INICIANDO EL TORNEO RAG (Modo: {modo_evaluacion}) ---")
    if limite_consultas:
        print(f"⚠️ Atención: La arena está limitada a {limite_consultas} consulta(s) de prueba.")
    print("Métricas a calcular: Recall@5, Recall@10, MAP@10, NDCG@10, Latencia\n")

    for cfg in PIPELINES_A_EVALUAR:
        nombre = cfg["nombre"]
        pipeline = PipelineRecuperacion(
            motor=motor,
            documentos_raw=documentos_bm25,
            base_retriever=cfg["base_retriever"],
            query_expansion=cfg["query_expansion"],
            post_processing=cfg["post_processing"]
        )
        
        # Evaluamos siempre con k=10 para poder calcular las métricas @10
        _, metricas = evaluador.evaluar_retrieval(
            funcion_busqueda=pipeline.invoke,
            estrategia_nombre=nombre,
            k=10,
            verbose=False, # Apagado para no ensuciar la consola con todas las métricas
            modo_evaluacion=modo_evaluacion,
            limite_consultas=limite_consultas
        )
        if metricas:
            resultados_arena.append(metricas)
        print(f"[OK] {nombre} evaluado.")

    # 5. Imprimir Tabla Comparativa Final
    if resultados_arena:
        df_arena = pd.DataFrame(resultados_arena)
        
        print("\n========================================================")
        print(f"--- RESULTADOS FINALES DE LA ARENA ({modo_evaluacion}) ---")
        print("========================================================")
        print(df_arena.to_markdown(index=False))
        
        # Guardar reporte consolidado
        out_dir = project_root / "data" / "03_output" / "evaluaciones"
        archivo_salida = out_dir / f"ARENA_RESULTADOS_{modo_evaluacion}.csv"
        df_arena.to_csv(archivo_salida, index=False)
        print(f"\n[INFO] Reporte cuantitativo guardado en: {archivo_salida}")
    elif modo_evaluacion == "human":
        print("\n[INFO] Modo auditoría humana finalizado. Los archivos Excel se guardaron en la carpeta de evaluaciones.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ejecuta la Arena de Evaluación RAG.")
    parser.add_argument("--modo", type=str, default="exact_match", choices=["exact_match", "human", "llm_judge"], 
                        help="Modo de evaluación: 'exact_match' (subcadena), 'human' (exportar a excel), 'llm_judge' (modelo semántico)")
    parser.add_argument("--limite", type=int, default=None, 
                        help="Límite de consultas a evaluar (ej. 1 para pruebas rápidas). Si no se provee, evalúa las 33 consultas.")
    
    args = parser.parse_args()
    
    correr_la_arena(modo_evaluacion=args.modo, limite_consultas=args.limite)
