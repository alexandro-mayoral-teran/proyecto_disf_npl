import sys
import os
import glob
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def exportar_auditoria_errores_b(subcarpeta_eval: str = "oficiales", run_folder: str = None):
    """
    Toma los resultados del análisis de errores desagregados (donde Categoría == B)
    y genera una plantilla Excel lista para la auditoría humana manual.
    """
    carpeta_base = project_root / "data" / "03_output" / "evaluaciones" / subcarpeta_eval
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
            print("No hay ejecuciones disponibles.")
            return
        ultimo_run = subcarpetas[0]
        
    print(f"Tomando datos de: {ultimo_run.name}")
    csvs = glob.glob(str(ultimo_run / "analisis_errores_desagregados_*.csv"))
    
    if not csvs:
        print("No se encontró el análisis de errores (Fase 3). Ejecuta 'evaluador_integral.py --fase3' primero.")
        return
        
    # Leer el primer archivo de análisis desagregado disponible (idealmente el Híbrido o SOTA)
    df = pd.read_csv(csvs[0])
    
    # Filtrar solo Errores B (Generación/Alucinación)
    df_errores_b = df[df["categoria_error"] == "B"]
    
    if len(df_errores_b) == 0:
        print("¡Excelente! No hubo Errores B de generación en la muestra.")
        return
        
    # Tomar 30 muestras o menos
    muestra_size = min(30, len(df_errores_b))
    df_muestra = df_errores_b.sample(n=muestra_size, random_state=42).copy()
    
    # Preparar el dataframe para humanos
    columnas_humanas = [
        "query_id", "pregunta", "categoria_error", "detalle_error", 
        "texto_esperado", "respuesta_generada", "modelo_extraccion_usado"
    ]
    df_export = df_muestra[columnas_humanas].copy()
    
    # Añadir columna vacía para que el usuario la llene
    df_export["Etiqueta_Humana"] = ""
    df_export["Comentarios"] = ""
    
    out_file = ultimo_run / "auditoria_manual_generacion.xlsx"
    
    try:
        df_export.to_excel(out_file, index=False)
        print(f"✅ Se exportó exitosamente una muestra de {muestra_size} errores B (Generación).")
        print(f"📁 Archivo: {out_file}")
        print("\nINSTRUCCIONES PARA LA AUDITORÍA:")
        print("Abre el archivo de Excel y llena la columna 'Etiqueta_Humana' con una de las siguientes opciones:")
        print(" - Útil")
        print(" - Parcial")
        print(" - Incorrecta")
        print(" - Alucinación")
        print(" - Refusal")
    except ImportError:
        print("⚠️ Falta la librería 'openpyxl'. Exportando a CSV...")
        out_file_csv = ultimo_run / "auditoria_manual_generacion.csv"
        df_export.to_csv(out_file_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Exportado a {out_file_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exportar Auditoría Manual")
    parser.add_argument("--carpeta", type=str, default="oficiales", help="Subcarpeta dentro de evaluaciones (ej. oficiales, pruebas_rapidas)")
    parser.add_argument("--run", type=str, default=None, help="Nombre exacto del run (ej. run_nube, run_20260606_085336). Si se omite, toma el más reciente.")
    args = parser.parse_args()
    
    exportar_auditoria_errores_b(args.carpeta, args.run)
