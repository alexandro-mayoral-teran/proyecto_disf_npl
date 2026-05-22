import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Añadir el directorio raíz para importar src
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.nlp_core.vectorizacion import MotorVectorizacion

def poblar_base_vectorial():
    print("Iniciando proceso de indexación de Markdowns en ChromaDB...")
    
    # 1. Definir rutas
    markdown_dir = project_root / "data" / "02_interim" / "markdown"
    if not markdown_dir.exists():
        print(f"❌ No se encontró la carpeta de Markdowns en {markdown_dir}")
        return

    archivos_md = list(markdown_dir.glob("*.md"))
    if not archivos_md:
        print(f"❌ No hay archivos .md en {markdown_dir}")
        return

    # 2. Inicializar Motor de Vectorización
    try:
        motor = MotorVectorizacion()
    except Exception as e:
        print(f"❌ Error al inicializar MotorVectorizacion: {e}")
        print("Asegúrate de tener OPENAI_API_KEY en tu archivo .env")
        return

    # 3. Iterar y Vectorizar cada archivo
    print(f"Se encontraron {len(archivos_md)} archivos Markdown. Comenzando vectorización...")
    for archivo in archivos_md:
        print(f"\n--- Procesando: {archivo.name} ---")
        try:
            # Por defecto usa MarkdownHeaderTextSplitter (EstrategiaChunking.ENCABEZADOS_MD)
            motor.indexar_documento_markdown(
                ruta_archivo=archivo, 
                collection_name="regulacion_disf"
            )
        except Exception as e:
            print(f"Error al indexar {archivo.name}: {e}")

    print("\n✅ ¡Proceso de vectorización completado!")
    
    # Imprimir conteo final
    try:
        total_docs = motor.vectorstore._collection.count()
        print(f"📦 Total de fragmentos (chunks) almacenados en ChromaDB: {total_docs}")
    except:
        pass

if __name__ == "__main__":
    poblar_base_vectorial()
