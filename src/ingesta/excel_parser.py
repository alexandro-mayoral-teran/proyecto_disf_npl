import pandas as pd
import os
import logging
from pathlib import Path

# Configuración básica de logging para registrar eventos en lugar de usar print
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extraer_tablas_excel_a_md(ruta_excel: str, ruta_salida_md: str) -> None:
    """
    Lee todas las pestañas de un archivo Excel y las consolida 
    en un solo documento Markdown.
    
    Args:
        ruta_excel (str): Ruta al archivo Excel de entrada (.xlsx).
        ruta_salida_md (str): Ruta donde se guardará el archivo Markdown de salida (.md).
        
    Returns:
        None
    """
    try:
        # 1. Cargar el archivo completo (ExcelFile permite ver los nombres de las hojas)
        archivo_xls = pd.ExcelFile(ruta_excel)
        nombres_hojas = archivo_xls.sheet_names
        
        # 2. Abrir el archivo .md en modo escritura
        with open(ruta_salida_md, 'w', encoding='utf-8') as archivo_md:
            nombre_base = os.path.basename(ruta_excel).replace('.xlsx', '')
            archivo_md.write(f"# Documento: {nombre_base}\n\n")
            
            # 3. Iterar sobre cada pestaña del Excel
            for hoja in nombres_hojas:
                # Leer la hoja actual
                df = pd.read_excel(ruta_excel, sheet_name=hoja)
                
                # Limpiar celdas vacías para evitar errores de parseo en Markdown
                df = df.fillna("") 
                
                # Escribir el título de la pestaña (como Subtítulo Markdown)
                archivo_md.write(f"## Tabla / Pestaña: {hoja}\n\n")
                
                # Convertir a tabla Markdown y escribir
                tabla_markdown = df.to_markdown(index=False)
                archivo_md.write(tabla_markdown + "\n\n")
                
        logger.info(f"Éxito: {len(nombres_hojas)} pestañas convertidas. Guardado en '{ruta_salida_md}'")
        
    except FileNotFoundError:
        logger.error(f"El archivo no fue encontrado en la ruta: {ruta_excel}")
    except ValueError as ve:
        logger.error(f"Error de valor (puede que el archivo no sea un Excel válido): {ve}")
    except Exception as e:
        # exception loggea el stacktrace automáticamente para debugear más fácil
        logger.exception(f"Error inesperado al procesar el archivo: {e}")

# --- Prueba en tu archivo real ---
# extraer_tablas_excel_a_md('catalogos_ml.xlsx', 'catalogos_ml.md')

def leer_excel_multipestana(path_excel: str, tipo: str) -> dict:
    """
    Función para leer cada pestaña de un excel
    path_excel : Ruta del archivo Excel.
    tipo : 'formulario' o 'catalogo', solo para metadatos.

    Devuelve un diccionario de dataframes donde cada una corresponde a una sección o un catálogo

    """
    path_excel = Path(path_excel)

    if not path_excel.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path_excel}")

    xls = pd.ExcelFile(path_excel)
    tablas = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(path_excel, sheet_name=sheet)

        # Limpieza básica de nombres de columnas
        df.columns = [
            str(c).strip().lower().replace(" ", "_")
            for c in df.columns
        ]

        # Eliminar filas completamente vacías
        df = df.dropna(how="all").reset_index(drop=True)

        # Guardar metadatos útiles
        df.attrs["archivo"] = path_excel.name
        df.attrs["tipo"] = tipo
        df.attrs["sheet"] = sheet

        tablas[sheet] = df

    return tablas

