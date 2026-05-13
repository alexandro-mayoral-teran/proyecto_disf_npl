from pathlib import Path
import sys

# Asegurar importaciones locales
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.utils.limpieza_texto import procesar_documento

##### Chunks por párrafo
def chunking_por_parrafo(texto: str, min_palabras: int = 20) -> list:
    chunks = []

    for bloque in re.split(r"\n\s*\n", texto):
        bloque = bloque.strip()
        n_palabras = len(re.findall(r"\b\w+\b", bloque))

        if n_palabras >= min_palabras:
            chunks.append(bloque)

    return chunks

##### Chunks fijos con overlap
def chunking_fijo_overlap(texto: str, chunk_size: int = 300, overlap: int = 50) -> list:
    palabras = re.findall(r"\b\w+\b", texto)
    chunks = []

    start = 0
    while start < len(palabras):
        end = start + chunk_size
        chunk = " ".join(palabras[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

##### Chunking estructural por títulos/artículos
def chunking_estructural(texto: str) -> list:
    patron = r"(?=(?:Artículo\s+\d+|TÍTULO\s+[A-ZÁÉÍÓÚÑ]+|CAPÍTULO\s+[A-ZÁÉÍÓÚÑ]+|Apartado\s+[A-Z]))"
    partes = re.split(patron, texto)

    chunks = []
    for parte in partes:
        parte = parte.strip()
        if len(parte.split()) >= 20:
            chunks.append(parte)

    return chunks

def normalizar_headers_regulatorios(texto: str) -> str:

    # TÍTULOS
    texto = re.sub(
        r'^(TÍTULO\s+[A-ZÁÉÍÓÚÑ]+.*)$',
        r'# \1',
        texto,
        flags=re.MULTILINE
    )
    texto = re.sub(
        r'^(Título\s+[A-ZÁÉÍÓÚÑ]+.*)$',
        r'# \1',
        texto,
        flags=re.MULTILINE
    )
    # ANEXOS
    texto = re.sub(
        r'^(ANEXO\s+[A-ZÁÉÍÓÚÑ]+.*)$',
        r'# \1',
        texto,
        flags=re.MULTILINE
    )
    texto = re.sub(
        r'^(Anexo\s+[A-ZÁÉÍÓÚÑ]+.*)$',
        r'# \1',
        texto,
        flags=re.MULTILINE
    )

    # CAPÍTULOS
    texto = re.sub(
        r'^(CAPÍTULO\s+[A-ZÁÉÍÓÚÑ]+.*)$',
        r'## \1',
        texto,
        flags=re.MULTILINE
    )
    texto = re.sub(
        r'^(Capítulo\s+[A-ZÁÉÍÓÚÑ]+.*)$',
        r'## \1',
        texto,
        flags=re.MULTILINE
    )

    # ARTÍCULOS
    texto = re.sub(
        r'^(ARTÍCULO\s+\d+.*)$',
        r'### \1',
        texto,
        flags=re.MULTILINE
    )
    texto = re.sub(
        r'^(Artículo\s+\d+.*)$',
        r'### \1',
        texto,
        flags=re.MULTILINE
    )

    return texto


def chunking_encabezados_md(texto: str, chunk_size: int = 300, overlap: int = 50) -> list:
#def crear_chunks_markdown(texto: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Lee un archivo Markdown, aplica limpieza de ruido OCR y fragmenta el texto
    basándose en la jerarquía de los encabezados (Títulos, Capítulos, Artículos).
    """

    texto = normalizar_headers_regulatorios(texto)

    # Definir los niveles de encabezados por los que queremos cortar.
    # Esto preservará el artículo o sección completa.
    headers_to_split_on = [
        ("#", "Header 1"),      # Ej: Títulos / Anexos
        ("##", "Header 2"),     # Ej: Capítulos
        ("###", "Header 3"),    # Ej: Secciones
        ("####", "Header 4"),   # Ej: Artículos
    ]

    # Inicializar el Text Splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Mantener el encabezado dentro del chunk es útil para dar contexto al LLM
    )

    # Dividir el texto
    documentos = markdown_splitter.split_text(texto)
    
    # dividir chunks gigantes manteniendo contexto
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks_finales = []

    for doc in documentos:

        subchunks = recursive_splitter.split_text(doc.page_content)

        for chunk in subchunks:
            chunks_finales.append(chunk)

            # chunks_finales.append({
            #     "texto": chunk,
            #     "metadata": doc.metadata
            # })
    
    return chunks_finales




if __name__ == "__main__":
    # --- PRUEBA RÁPIDA ---
    archivo_prueba = project_root / "data" / "02_interim"/ "CUB_extracto.md"
    
    if archivo_prueba.exists():
        print(f"Fragmentando archivo: {archivo_prueba.name}...")
        
        # Probamos el chunking
        chunks_generados = chunking_encabezados_md(archivo_prueba, origen="CNBV")
        
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
