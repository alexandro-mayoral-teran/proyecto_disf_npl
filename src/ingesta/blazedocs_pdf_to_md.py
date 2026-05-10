import time
import requests
from pathlib import Path
import os
import sys
import json
import pandas as pd

from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()

BLAZEDOCS_API_URL = "https://blazedocs.io/api/v1/convert"

def get_blazedocs_api_key() -> str:
    """
    Se toma la API key de BlazeDocs desde una variable de ambiente
    """
    api_key = os.getenv("BLAZEDOCS_API_KEY")

    if not api_key:
        raise ValueError(
            "No se encontró la API key de BlazeDocs en el archivo .env")

    return api_key

def convert_pdf_to_markdown(pdf_path: Path) -> dict:
    """
    pdf_path : Path al archivo PDF
    devuelve un diccionario con el texto markdown y los metadatos que regresa la API.
        Path to the PDF file.
    """
    api_key = get_blazedocs_api_key()


    headers = {
        "Authorization": f"Bearer {api_key}"}

    with open(pdf_path, "rb") as file:
        files = {
            "file": (pdf_path.name, file, "application/pdf")

        }


        response = requests.post(
            BLAZEDOCS_API_URL,
            headers=headers,
            files=files
        )

    # Error si falla el request HTTP
    response.raise_for_status()

    result = response.json()

    # Indica si se realizó la conversión correctamente o si se generó un error
    if not result.get("success", False):
        raise ValueError(
            f"BlazeDocs conversion failed for {pdf_path.name}: "
            f"{result.get('error', 'Unknown error')}"
        )

    return result

def save_markdown(markdown_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(markdown_text)

def convert_pdf_folder_to_markdown(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    pause_seconds: float = 1.0
) -> pd.DataFrame:
    """
    Convierte los documentos encontrados en una carpeta en Markdown.

    input_dir : Directorio donde se encuentran los PDFs.
    output_dir : Directorio donde se guardan los archivos convertidos.
    overwrite : booleano. Si es falso, los archivos ya convertidos encontrados, no se convierten de nuevo.
    pause_seconds : float. Segundos de espera entre llamadas a la API.

    Devuelve un pd.DataFrame con los registros de conversión por cada PDF encontrado.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files were found in: {input_dir}")

    conversion_log = []

    for pdf_path in pdf_files:
        output_path = output_dir / f"{pdf_path.stem}.md"

        if output_path.exists() and not overwrite:
            conversion_log.append({
                "file_name": pdf_path.name,
                "output_file": output_path.name,
                "status": "skipped_already_exists",
                "page_count": None,
                #"token_count": None,
                "processing_time_ms": None,
                "error": None
            })
            continue

        try:
            result = convert_pdf_to_markdown(pdf_path)

            data = result["data"]
            markdown_text = data["markdown"]

            save_markdown(markdown_text, output_path)

            conversion_log.append({
                "file_name": pdf_path.name,
                "output_file": output_path.name,
                "status": "success",
                "page_count": data.get("page_count"),
                "token_count": data.get("token_count"),
                "processing_time_ms": data.get("processing_time_ms"),
                "error": None
            })

            time.sleep(pause_seconds)

        except Exception as e:
            conversion_log.append({
                "file_name": pdf_path.name,
                "output_file": output_path.name,
                "status": "error",
                "page_count": None,
                "token_count": None,
                "processing_time_ms": None,
                "error": str(e)
            })

    return pd.DataFrame(conversion_log)

if __name__ == "__main__":
    # --- BLOQUE DE PRUEBA RÁPIDA ---
    # Esto solo se ejecutará si corres el archivo directamente: python src/ingesta/blazedocs_pdf_to_md.py
    
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Asegurar que el root esté en sys.path para importaciones locales
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        
    print(f"Iniciando prueba de conversión.")
    print(f"Raíz del proyecto detectada: {project_root}")
    
    # Configurar las rutas para la prueba usando tu estructura
    input_pdf_dir = project_root / "data" / "01_raw" / "pdfs"
    output_md_dir = project_root / "data" / "02_interim" / "markdown"
    
    # Crear el directorio de salida si no existe
    output_md_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nBuscando PDFs en: {input_pdf_dir}")
    print(f"Guardando resultados en: {output_md_dir}")
    
    try:
        # Iniciamos la conversión (con overwrite=False para no gastar cuota en los que ya existan)
        df_log = convert_pdf_folder_to_markdown(
            input_dir=input_pdf_dir,
            output_dir=output_md_dir,
            overwrite=False,
            pause_seconds=1.0
        )
        print("\n=== Resultados de Conversión ===")
        print(df_log)
    except FileNotFoundError as e:
        print(f"\n[Error]: {e}")
        print("Asegúrate de que haya archivos PDF en la carpeta de entrada.")
    except Exception as e:
        print(f"\n[Error Inesperado]: {e}")


