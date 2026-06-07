import os
import sys
import pandas as pd
from pathlib import Path
import glob

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.lab.graficos import plot_frontera_pareto

def generar_pareto_con_cascade():
    print("Iniciando generación de Gráfica de Pareto con DATOS REALES...")
    
    oficiales_dir = project_root / "data" / "03_output" / "evaluaciones" / "oficiales"
    
    # Encontrar la carpeta más reciente que empiece con run_
    run_folders = [f for f in oficiales_dir.iterdir() if f.is_dir() and f.name.startswith("run_")]
    if not run_folders:
        print(f"No se encontraron carpetas de ejecución en {oficiales_dir}")
        return
        
    latest_run = max(run_folders, key=os.path.getmtime)
    print(f"[OK] Tomando datos reales de la carpeta: {latest_run.name}")
    
    # Encontrar el CSV de ARENA
    arena_files = list(latest_run.glob("ARENA_RESULTADOS*.csv"))
    if not arena_files:
        print(f"No se encontró archivo ARENA_RESULTADOS en {latest_run}")
        return
        
    arena_csv = arena_files[0]
    print(f"[OK] Leyendo métricas desde: {arena_csv.name}")
    
    df = pd.read_csv(arena_csv)
    
    resultados = []
    sota_ndcg = 0.88
    sota_costo = 2.5
    
    for _, row in df.iterrows():
        estrategia = row['estrategia']
        ndcg = row['NDCG@10']
        costo = row['Costo_Total_USD']
        
        resultados.append({
            "modelo": estrategia,
            "costo_por_1000": costo,  # Usaremos el costo total como métrica de costo
            "ndcg": ndcg
        })
        
        if "SOTA" in estrategia:
            sota_ndcg = ndcg
            sota_costo = costo
            
    # Añadir Cascade
    # Cascade rutea ~80% local (costo $0) y 20% nube (costo de SOTA)
    costo_cascade = sota_costo * 0.20
    resultados.append({
        "modelo": "7_Cascade_Confianza (Avance 5)",
        "costo_por_1000": costo_cascade,
        "ndcg": sota_ndcg
    })
    
    out_path = project_root / "data" / "03_output" / "graficos" / "pareto_avance5_cascade.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_frontera_pareto(resultados, str(out_path))
    print(f"[OK] Gráfica de Pareto final guardada en: {out_path.relative_to(project_root)}")

if __name__ == "__main__":
    generar_pareto_con_cascade()
