import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

def calcular_diversidad(ruta_csv_modelo_a: str, ruta_csv_modelo_b: str, nombre_a: str, nombre_b: str, ruta_salida: str = None):
    """
    Calcula las métricas de diversidad (ENS-D) entre dos modelos basándose en sus resultados a nivel consulta.
    Compara si ambos acertaron (hit > 0), ambos fallaron, o si discreparon.
    """
    print(f"Calculando Diversidad: {nombre_a} vs {nombre_b}")
    
    df_a = pd.read_csv(ruta_csv_modelo_a)
    df_b = pd.read_csv(ruta_csv_modelo_b)
    
    # Asegurarnos de que están alineados por query_id
    if not 'query_id' in df_a.columns or not 'query_id' in df_b.columns:
        raise ValueError("Los CSV deben contener la columna 'query_id'")
        
    df_merge = pd.merge(df_a, df_b, on='query_id', suffixes=('_A', '_B'))
    total_queries = len(df_merge)
    
    if total_queries == 0:
        print("No hay consultas en común para comparar.")
        return
        
    # Definimos como "éxito" si hit == 1 o ndcg_10 > 0
    # Asumimos la columna 'hit' como binaria (1 acertó, 0 falló)
    if 'hit_A' not in df_merge.columns:
        # Fallback a ndcg si no hay hit
        exito_A = df_merge['ndcg_10_A'] > 0
        exito_B = df_merge['ndcg_10_B'] > 0
    else:
        exito_A = df_merge['hit_A'] > 0
        exito_B = df_merge['hit_B'] > 0

    # Cálculos de Acuerdo / Desacuerdo
    ambos_aciertan = ((exito_A == True) & (exito_B == True)).sum()
    ambos_fallan = ((exito_A == False) & (exito_B == False)).sum()
    a_acierta_b_falla = ((exito_A == True) & (exito_B == False)).sum()
    a_falla_b_acierta = ((exito_A == False) & (exito_B == True)).sum()
    
    disagreement = a_acierta_b_falla + a_falla_b_acierta
    disagreement_rate = disagreement / total_queries
    
    # Correlación de Errores (Phi Coefficient o correlación de Pearson sobre variables binarias)
    # 1 = Error (Falló), 0 = No Error (Acertó)
    errores_A = (~exito_A).astype(int)
    errores_B = (~exito_B).astype(int)
    correlacion_errores = errores_A.corr(errores_B)
    
    # Oracle: Cuánto acertaríamos si tuviéramos un Oráculo que escoge al modelo correcto siempre que al menos uno acierte
    oracle_hits = ambos_aciertan + disagreement
    oracle_accuracy = oracle_hits / total_queries
    mejor_accuracy_individual = max(exito_A.mean(), exito_B.mean())
    oracle_gap = oracle_accuracy - mejor_accuracy_individual
    
    resumen = {
        "modelo_A": nombre_a,
        "modelo_B": nombre_b,
        "total_queries": int(total_queries),
        "matriz_acuerdo": {
            "ambos_aciertan": int(ambos_aciertan),
            "ambos_fallan": int(ambos_fallan),
            f"solo_{nombre_a}_acierta": int(a_acierta_b_falla),
            f"solo_{nombre_b}_acierta": int(a_falla_b_acierta)
        },
        "metricas_ens_d": {
            "disagreement_rate": float(disagreement_rate),
            "error_correlation": float(correlacion_errores) if not pd.isna(correlacion_errores) else 1.0,
            "oracle_accuracy": float(oracle_accuracy),
            "oracle_gap": float(oracle_gap)
        }
    }
    
    # Imprimir Reporte
    print("-" * 50)
    print(f"Disagreement Rate: {disagreement_rate*100:.2f}% (Tasa de discrepancia)")
    print(f"Correlacion de Errores: {correlacion_errores:.4f} (Si >0.8, el ensamble aporta poco)")
    print(f"Oracle Gap: +{oracle_gap*100:.2f}% (Maximo Lift teorico si los ensamblamos)")
    print("-" * 50)
    print(f"Ambos aciertan: {ambos_aciertan}")
    print(f"Ambos fallan : {ambos_fallan}")
    print(f"Solo {nombre_a} acierta: {a_acierta_b_falla}")
    print(f"Solo {nombre_b} acierta: {a_falla_b_acierta}")
    
    if ruta_salida:
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resumen, f, indent=4, ensure_ascii=False)
        print(f"Reporte guardado en {ruta_salida}")
        
    return resumen

if __name__ == "__main__":
    # Uso de ejemplo (requiere correr el evaluador exhaustivo primero para tener los CSV)
    print("Este script calculará la diversidad entre dos modelos una vez que tengas los CSV generados.")
    print("En el dashboard de Streamlit, podrás seleccionar qué CSVs comparar dinámicamente.")
