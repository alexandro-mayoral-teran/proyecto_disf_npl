import sys
import argparse
from pathlib import Path

# Añadir el directorio raíz para importar src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# python src/lab/convertir_pdf.py "data/01_raw/pdfs/CUB_completo.pdf"

from src.ingesta.ingestor import IngestorDocumentos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte un documento a MD usando el Ingestor de DISF")
    parser.add_argument("input_path", help="Ruta al archivo a convertir (ej. data/01_raw/Anexo_33_CUB.pdf)")
    args = parser.parse_args()

    input_file = Path(args.input_path)
    output_dir = project_root / "data" / "02_interim" / "markdown"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("==================================================")
    print(f"🔄 CONVIRTIENDO: {input_file.name}")
    print("==================================================")
    
    ingestor = IngestorDocumentos(output_dir=output_dir)
    res = ingestor.procesar_archivo(input_file)
    
    if res["status"] == "success":
        print(f"✅ ¡Éxito! Archivo generado: {res['output_file']}")
        print(f"Puedes encontrarlo en: {output_dir}")
    elif res["status"] == "skipped_already_exists":
        print(f"⏭️ El archivo Markdown ya existe en la carpeta.")
    else:
        print(f"❌ Ocurrió un error: {res.get('error')}")
