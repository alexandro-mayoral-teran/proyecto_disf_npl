import sys
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def calcular_delta_bootstrap(scores_baseline_e3: list, scores_actual: list, num_resamples=1000):
    if len(scores_baseline_e3) != len(scores_actual):
        print("Error: Las listas de scores deben tener la misma longitud para Paired Bootstrap.")
        return 0, 0, 0
        
    random.seed(42)
    n = len(scores_actual)
    
    media_baseline = sum(scores_baseline_e3) / n
    media_actual = sum(scores_actual) / n
    delta_original = media_actual - media_baseline
    
    deltas_bootstrap = []
    
    for _ in range(num_resamples):
        indices = random.choices(range(n), k=n)
        muestra_base = [scores_baseline_e3[i] for i in indices]
        muestra_act = [scores_actual[i] for i in indices]
        delta_b = (sum(muestra_act) / n) - (sum(muestra_base) / n)
        deltas_bootstrap.append(delta_b)
        
    deltas_bootstrap.sort()
    lim_inf = deltas_bootstrap[int(0.025 * num_resamples)]
    lim_sup = deltas_bootstrap[int(0.975 * num_resamples)]
    
    return delta_original, lim_inf, lim_sup

def ejecutar_demostracion_ma4():
    print("Calculando Lift Estadísticamente Significativo (MA4 E5) con DATOS REALES...")
    
    oficiales_dir = project_root / "data" / "03_output" / "evaluaciones" / "oficiales"
    
    run_folders = [f for f in oficiales_dir.iterdir() if f.is_dir() and f.name.startswith("run_")]
    if not run_folders:
        print(f"No se encontraron carpetas de ejecución en {oficiales_dir}")
        return
        
    latest_run = max(run_folders, key=os.path.getmtime)
    print(f"[OK] Tomando datos reales de la carpeta: {latest_run.name}")
    
    # Buscar los archivos de resultados individuales
    baseline_files = list(latest_run.glob("*_2_Baseline_Sem*ntico*.csv"))
    sota_files = list(latest_run.glob("*_6_SOTA_Completo*.csv"))
    
    if not baseline_files or not sota_files:
        print(f"Error: No se encontraron los archivos CSV individuales para el Baseline Semántico y/o SOTA Completo en {latest_run}")
        return
        
    baseline_csv = baseline_files[0]
    sota_csv = sota_files[0]
    
    print(f"[OK] Leyendo E3-BL5 (Baseline) desde: {baseline_csv.name}")
    print(f"[OK] Leyendo Cascade (SOTA) desde: {sota_csv.name}")
    
    df_base = pd.read_csv(baseline_csv)
    df_sota = pd.read_csv(sota_csv)
    
    if 'ndcg_10' not in df_base.columns or 'ndcg_10' not in df_sota.columns:
        print("Error: Los archivos no contienen la columna 'ndcg_10'")
        return
        
    scores_e3 = df_base['ndcg_10'].fillna(0).tolist()
    scores_e5 = df_sota['ndcg_10'].fillna(0).tolist()
    
    n = len(scores_e3)
    
    delta, ci_inf, ci_sup = calcular_delta_bootstrap(scores_e3, scores_e5, num_resamples=1000)
    
    print(f"\n=========================================")
    print(f" COMPARATIVA: Cascade (Avance 5) vs E3-BL5")
    print(f" Consultas analizadas: {n}")
    print(f" NDCG@10 Promedio E3-BL5: {sum(scores_e3)/n:.4f}")
    print(f" NDCG@10 Promedio Cascade: {sum(scores_e5)/n:.4f}")
    print(f"-----------------------------------------")
    print(f" Delta NDCG (Lift Promedio): +{delta:.4f}")
    print(f" CI 95% del Delta: [{ci_inf:.4f}, {ci_sup:.4f}]")
    print(f"=========================================\n")
    
    if ci_inf > 0:
        print("[OK] CONCLUSIÓN: El modelo actual supera estadísticamente al E3-BL5.")
        print("Cumple con el criterio (b) de ENS-F de la rúbrica E5.")
    else:
        print("[WARN] CONCLUSIÓN: El intervalo de confianza cruza el cero. El lift no es significativo.")

if __name__ == "__main__":
    ejecutar_demostracion_ma4()
