from pathlib import Path
import sys

# Asegurar importaciones locales
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.utils.limpieza_texto import procesar_documento

def crear_chunks_markdown(ruta_archivo: Path, origen: str = "CNBV"):
    """
    Lee un archivo Markdown, aplica limpieza de ruido OCR y fragmenta el texto
    basándose en la jerarquía de los encabezados (Títulos, Capítulos, Artículos).
    """
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        texto_md = f.read()

    # 1. Limpieza inicial (quitar números de página, saltos innecesarios, etc.)
    texto_limpio = procesar_documento(texto_md, origen=origen)

    # 2. Definir los niveles de encabezados por los que queremos cortar.
    # Esto preservará el artículo o sección completa.
    headers_to_split_on = [
        ("#", "Header 1"),      # Ej: Títulos / Anexos
        ("##", "Header 2"),     # Ej: Capítulos
        ("###", "Header 3"),    # Ej: Secciones
        ("####", "Header 4"),   # Ej: Artículos
    ]

    # 3. Inicializar el Text Splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Mantener el encabezado dentro del chunk es útil para dar contexto al LLM
    )

    # 4. Dividir el texto
    chunks = markdown_splitter.split_text(texto_limpio)
    
    return chunks

if __name__ == "__main__":
    # --- PRUEBA RÁPIDA ---
    archivo_prueba = project_root / "data" / "02_interim"/ "CUB_extracto.md"
    
    if archivo_prueba.exists():
        print(f"Fragmentando archivo: {archivo_prueba.name}...")
        
        # Probamos el chunking
        chunks_generados = crear_chunks_markdown(archivo_prueba, origen="CNBV")
        
        print(f"Total de chunks (fragmentos) generados: {len(chunks_generados)}\n")
        
        if chunks_generados:
            print("="*50)
            print("--- EJEMPLO DEL PRIMER CHUNK ---")
            print(f"Metadatos extraídos de los títulos: {chunks_generados[0].metadata}")
            print(f"Contenido (primeros 400 caracteres):\n{chunks_generados[0].page_content[:400]}...")
            
            print("\n" + "="*50)
            print("--- EJEMPLO DEL ÚLTIMO CHUNK ---")
            print(f"Metadatos extraídos de los títulos: {chunks_generados[-1].metadata}")
            print(f"Contenido (primeros 400 caracteres):\n{chunks_generados[-1].page_content[:400]}...")
    else:
        print(f"No se encontró el archivo de prueba en {archivo_prueba}")
        print("Intenta con el nombre de otro documento Markdown que sí exista.")
