import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Añadir el directorio raíz para importar src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Forzar salida estándar en UTF-8 para evitar errores de impresión con emojis en Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(project_root / ".env")

from src.nlp_core.retrieval import MotorBusqueda
from src.nlp_core.evals.evaluador import EvaluadorRAG
from src.nlp_core.pipeline import PipelineRecuperacion
from src.nlp_core.generacion import extraer_rag_simple
from langchain_core.documents import Document

from src.nlp_core.config_llm import get_llm_model_name

def recolectar_metadatos_llm():
    return {
        "LLM_Modelo_QA": get_llm_model_name("qa"),
        "Es_QA_Local": os.getenv("USE_LOCAL_QA", "false").lower() == "true",
        "LLM_Modelo_Extraccion": get_llm_model_name("extraction"),
        "Es_Extraccion_Local": os.getenv("USE_LOCAL_EXTRACTION", "false").lower() == "true",
        "Es_Juez_Local": os.getenv("USE_LOCAL_JUDGE", "false").lower() == "true",
        "Es_Expansion_Local": os.getenv("USE_LOCAL_EXPANSION", "true").lower() == "true"
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluador Integral RAG")
    parser.add_argument("--rapido", action="store_true", help="Ejecuta en modo desarrollo (1 consulta, carpeta pruebas_rapidas)")
    parser.add_argument("--exhaustivo", action="store_true", help="Ejecuta todas las consultas (carpeta oficiales)")
    parser.add_argument("--fase1", action="store_true", help="Ejecutar SOLO Fase 1: Arena de Retrieval")
    parser.add_argument("--fase2", action="store_true", help="Ejecutar SOLO Fase 2: Prueba Ciega")
    parser.add_argument("--fase3", action="store_true", help="Ejecutar SOLO Fase 3: Análisis Desagregado (Tarda más)")
    args = parser.parse_args()

    ejecutar_todas = not (args.fase1 or args.fase2 or args.fase3)
    run_fase1 = ejecutar_todas or args.fase1
    run_fase2 = ejecutar_todas or args.fase2
    run_fase3 = ejecutar_todas or args.fase3

    # Routing de carpetas
    if args.rapido:
        subcarpeta = "pruebas_rapidas"
        limite_consultas = 1
        print("⚡ MODO RÁPIDO ACTIVADO (1 consulta por fase)")
    else:
        subcarpeta = "oficiales"
        limite_consultas = None
        print("🚀 MODO EXHAUSTIVO ACTIVADO (Todas las consultas)")

    metadatos = recolectar_metadatos_llm()
    print("\n--- Metadatos de la Corrida ---")
    for k, v in metadatos.items():
        print(f" - {k}: {v}")
    
    # 1. Inicializar el Evaluador
    gt_path = project_root / "data" / "evaluacion_dataset.json"
    evaluador = EvaluadorRAG(ground_truth_path=gt_path, subcarpeta_salida=subcarpeta)
    
    # 2. Inicializar Motor de Búsqueda
    try:
        motor = MotorBusqueda()
        if motor.vectorstore._collection.count() == 0:
            print("⚠️ ADVERTENCIA: Tu ChromaDB está vacío. Necesitas poblarlo primero.")
            return
    except Exception as e:
        print(f"❌ Error al inicializar MotorBusqueda: {e}")
        return

    # Extraer corpus para BM25
    raw_data = motor.vectorstore.get(include=["metadatas", "documents"])
    documentos_bm25 = [
        Document(page_content=txt, metadata=meta) 
        for txt, meta in zip(raw_data["documents"], raw_data["metadatas"])
    ]

    # --- FASE 1: ARENA DE RETRIEVAL ---
    if run_fase1:
        print("\n========================================================")
        print("🏟️  FASE 1: ARENA DE RETRIEVAL")
        print("========================================================")
    
        import json
        config_path = project_root / "data" / "config_experimentos.json"
        if not config_path.exists():
            print(f"❌ Error: No se encontró el archivo de configuración en {config_path}")
            return
            
        with open(config_path, "r", encoding="utf-8") as f:
            config_experimentos = json.load(f)
            
        PIPELINES_A_EVALUAR = config_experimentos["pruebas_rapidas"] if args.rapido else config_experimentos["exhaustivos"]

        resultados_arena = []
        for cfg in PIPELINES_A_EVALUAR:
            nombre = cfg["nombre"]
            pipeline = PipelineRecuperacion(
                motor=motor,
                documentos_raw=documentos_bm25,
                base_retriever=cfg["base_retriever"],
                query_expansion=cfg["query_expansion"],
                post_processing=cfg["post_processing"]
            )
            
            _, metricas = evaluador.evaluar_retrieval(
                funcion_busqueda=pipeline.invoke,
                estrategia_nombre=nombre,
                k=10,
                verbose=False,
                modo_evaluacion="llm_judge",
                limite_consultas=limite_consultas,
                metadatos_llm=metadatos
            )
            if metricas:
                resultados_arena.append(metricas)
            print(f"[OK] {nombre} evaluado.")

        if resultados_arena:
            df_arena = pd.DataFrame(resultados_arena)
            archivo_salida = evaluador.out_dir / f"ARENA_RESULTADOS_LLM_JUDGE_{evaluador.run_timestamp}.csv"
            df_arena.to_csv(archivo_salida, index=False)
            print(f"📊 Arena guardada en: {archivo_salida.name}")

    # --- FASE 2: DATA CONTAMINATION CHECK ---
    if run_fase2:
        print("\n========================================================")
        print("🕵️  FASE 2: DATA CONTAMINATION CHECK (PRUEBA CIEGA)")
        print("========================================================")
    
        modelo_qa_actual = metadatos["LLM_Modelo_QA"]
        res_ciego = evaluador.evaluar_contaminacion_ciega(
            candidato_nombre=f"BlindTest_{modelo_qa_actual}",
            task="qa",
            limite_consultas=limite_consultas,
            verbose=False,
            metadatos_llm=metadatos
        )
        print(f"✅ Prueba Ciega completada para {modelo_qa_actual}. Precisión de memoria: {res_ciego['Precision_Ciega_Porcentaje']}%")

    # --- FASE 3: ANÁLISIS DE ERRORES DESAGREGADO ---
    if run_fase3:
        print("\n========================================================")
        print("🧩 FASE 3: ANÁLISIS DE ERRORES DESAGREGADO (A, B, C)")
        print("========================================================")
    
        pipeline_hibrido = PipelineRecuperacion(
            motor=motor,
            documentos_raw=documentos_bm25,
            base_retriever="hibrido",
            query_expansion=None,
            post_processing=None
        )
        
        def funcion_busqueda_wrapper(query, k):
            return pipeline_hibrido.invoke(query, k=k)
            
        def funcion_qa_wrapper(query, k):
            return extraer_rag_simple(query, k=k)
            
        conteos, _ = evaluador.evaluar_desagregacion_errores(
            funcion_busqueda=funcion_busqueda_wrapper,
            funcion_qa_extraccion=funcion_qa_wrapper,
            estrategia_nombre="Hibrido_Extraccion_Completa",
            limite_consultas=limite_consultas,
            verbose=False,
            metadatos_llm=metadatos
        )
        
        print(f"✅ Análisis desagregado completado. Errores A (Retrieval): {conteos['Fallo_Retrieval_A']}, B (Generación): {conteos['Fallo_Generacion_B']}, C (Formato): {conteos['Fallo_Formato_C']}")

    print(f"\n🚀 EJECUCIÓN COMPLETADA. Resultados guardados en {evaluador.out_dir.relative_to(project_root)}")

if __name__ == "__main__":
    main()
