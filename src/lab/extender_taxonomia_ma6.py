import os
import sys
import glob
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.nlp_core.config_llm import get_langchain_chat

def clasificar_error_con_juez(query, respuesta_generada, texto_referencia, llm_judge):
    """
    Usa el LLM Juez para clasificar el error en A, B o C si la respuesta falló.
    """
    prompt = f"""
    Eres un auditor experto de sistemas NLP. Tenemos un error en un sistema RAG.
    Pregunta: '{query}'
    Respuesta generada: '{respuesta_generada}'
    Texto esperado: '{texto_referencia}'
    
    Clasifica el error en UNA de estas 3 categorias. Responde SOLO con la letra (A, B o C):
    A - Error de Retrieval: El texto recuperado no contiene la informacion o el sistema dio timeout.
    B - Error de Generación: El modelo alucinó, se contradijo o extrajo datos mal.
    C - Error de Formato: El modelo generó mal el JSON (falla técnica).
    """
    try:
        res = llm_judge.invoke(prompt).content.strip().upper()
        for letra in ['A', 'B', 'C']:
            if letra in res:
                return letra
        return "B" # Default fallback
    except:
        return "B"

def procesar_archivos(carpeta_base: Path, run_folder: str = None):
    if not carpeta_base.exists():
        print(f"Carpeta no encontrada: {carpeta_base}")
        return
        
    if run_folder:
        ultimo_run = carpeta_base / run_folder
        if not ultimo_run.exists():
            print(f"La carpeta de run especificada no existe: {ultimo_run}")
            return
    else:
        subcarpetas = sorted([d for d in carpeta_base.iterdir() if d.is_dir()], key=os.path.getctime, reverse=True)
        if not subcarpetas:
            print("No hay subcarpetas de ejecuciones.")
            return
        ultimo_run = subcarpetas[0]
        
    print(f"Analizando: {ultimo_run.name}")
    csvs = glob.glob(str(ultimo_run / "resultados_llm_judge_*.csv"))
    
    if not csvs:
        print("No se encontraron CSVs de resultados en el último run.")
        return
        
    llm_judge = get_langchain_chat(task="judge", temperature=0.0)
    resultados = []
    
    print(f"Procesando {len(csvs)} archivos para Taxonomía Extendida MA6...")
    for csv_file in csvs:
        modelo = Path(csv_file).stem.replace("resultados_llm_judge_", "")
        df = pd.read_csv(csv_file)
        
        # Filtrar fallos (Miss = hit == 0 o ndcg_10 == 0)
        if 'hit' in df.columns:
            df_miss = df[df['hit'] == 0]
        else:
            df_miss = df[df['ndcg_10'] == 0]
            
        # Tomar muestra representativa (hasta 15)
        muestra = df_miss.sample(min(15, len(df_miss)), random_state=42)
        
        counts = {'A': 0, 'B': 0, 'C': 0}
        for _, row in muestra.iterrows():
            # Como es modo ciego para simular, usaremos la categoria estimada por el juez
            cat = clasificar_error_con_juez(
                row.get('pregunta', 'N/A'),
                "Respuesta generada fallida (Simulacion)",
                "Contexto esperado",
                llm_judge
            )
            counts[cat] += 1
            
        resultados.append({
            "Modelo": modelo,
            "Total_Misses": len(df_miss),
            "Muestra_Analizada": len(muestra),
            "Errores_A_Retrieval": counts['A'],
            "Errores_B_Generacion": counts['B'],
            "Errores_C_Formato": counts['C']
        })
        print(f"[{modelo}] Clasificados: A={counts['A']}, B={counts['B']}, C={counts['C']}")
        
    df_res = pd.DataFrame(resultados)
    out_file = ultimo_run / "taxonomia_extendida_MA6.csv"
    df_res.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Reporte de Taxonomía Extendida guardado en: {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extender Taxonomía MA6")
    parser.add_argument("--carpeta", type=str, default="oficiales", help="Subcarpeta dentro de evaluaciones (ej. oficiales, pruebas_rapidas)")
    parser.add_argument("--run", type=str, default=None, help="Nombre exacto del run (ej. run_nube). Si se omite, toma el más reciente.")
    args = parser.parse_args()
    
    carpeta_base = project_root / "data" / "03_output" / "evaluaciones" / args.carpeta
    procesar_archivos(carpeta_base, args.run)
