import os
import sys
from pathlib import Path

# Añadir el directorio raíz para importar src
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# python src/lab/poblar_desde_manifest.py

from src.nlp_core.vectorizacion import indexar_desde_manifest
from src.nlp_core.chunking import ContextualizadorLLM, RegulacionChunker, EstrategiaChunking
from dotenv import load_dotenv

if __name__ == "__main__":
    # Cargar variables de entorno (OPENAI_API_KEY, etc)
    load_dotenv()
    
    # Rutas absolutas hacia el proyecto principal
    manifest_path = project_root / "data" / "01_raw" / "manifest.yaml"
    base_dir = project_root / "data" / "02_interim" / "markdown"
    
    print("==================================================")
    print("🚀 INICIANDO CONSTRUCCIÓN DE BASE VECTORIAL GOBERNADA")
    print("==================================================")
    print(f"Leyendo manifest desde: {manifest_path}")
    print(f"Buscando archivos Markdown en: {base_dir}")
    print("--------------------------------------------------")
    
    # Mantenemos tu pipeline avanzado de RAG
    chunker_seleccionado = RegulacionChunker(EstrategiaChunking.ENCABEZADOS_MD)
    postprocesadores_seleccionados = [ContextualizadorLLM()]
    
    try:
        vectorstore = indexar_desde_manifest(
            manifest_path=manifest_path,
            base_dir=base_dir,
            collection_name="regulacion_disf",
            chunker=chunker_seleccionado,
            postprocesadores=postprocesadores_seleccionados
        )
        
        if vectorstore:
            print("--------------------------------------------------")
            print("✅ ¡Indexación completada exitosamente!")
            print("Tu base ChromaDB ha sido creada y está lista para recibir consultas.")
        else:
            print("--------------------------------------------------")
            print("⚠️ No se generó la base. Revisa que tu manifest.yaml tenga archivos válidos.")
    except Exception as e:
        print(f"❌ Ocurrió un error fatal durante la indexación: {str(e)}")

