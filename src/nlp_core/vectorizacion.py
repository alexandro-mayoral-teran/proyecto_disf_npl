import os
import sys
from pathlib import Path

# Asegurar importaciones locales
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.nlp_core.chunking import crear_chunks_markdown

# Cargar variables de entorno (OPENAI_API_KEY)
load_dotenv()

# Directorio donde se guardará la base de datos vectorial localmente
CHROMA_PERSIST_DIR = str(project_root / "data" / "03_output" / "chroma_db")

def obtener_embeddings():
    """
    Inicializa el modelo de embeddings de OpenAI.
    Usamos text-embedding-3-small porque es rápido, muy barato y tiene excelente rendimiento.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")
        
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

def indexar_documento(ruta_archivo: Path, origen: str = "CNBV", collection_name: str = "regulacion_disf"):
    """
    Toma un archivo Markdown, lo fragmenta usando nuestra estrategia estructural
    y guarda los chunks en ChromaDB.
    """
    print(f"1. Fragmentando el documento: {ruta_archivo.name}...")
    chunks = crear_chunks_markdown(ruta_archivo, origen=origen)
    print(f"   Se generaron {len(chunks)} chunks.")
    
    # Asegurar que cada chunk tenga el nombre del documento en sus metadatos
    for chunk in chunks:
        chunk.metadata["source_file"] = ruta_archivo.name
        
    print("2. Inicializando modelo de Embeddings y ChromaDB...")
    embeddings = obtener_embeddings()
    
    # Creamos o actualizamos la base de datos vectorial
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=collection_name
    )
    
    print(f"3. ¡Documento indexado con éxito en {CHROMA_PERSIST_DIR}!")
    return vectorstore

def buscar_similitud(query: str, collection_name: str = "regulacion_disf", k: int = 3):
    """
    Realiza una búsqueda de similitud en la base de datos vectorial.
    Devuelve los 'k' fragmentos más relevantes para la consulta.
    """
    embeddings = obtener_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR, 
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    print(f"Buscando: '{query}'...")
    resultados = vectorstore.similarity_search(query, k=k)
    return resultados

if __name__ == "__main__":
    # --- PRUEBA RÁPIDA DE VECTORIZACIÓN Y BÚSQUEDA ---
    
    # 1. Archivo que vamos a indexar
    archivo_prueba = project_root / "data" / "02_interim" / "CUB_extracto.md"

    if archivo_prueba.exists():
        print("=== INICIANDO PRUEBA DE VECTORIZACIÓN ===")
        # Indexar
        vectorstore = indexar_documento(archivo_prueba, origen="CNBV")
        
        print("\n=== PRUEBA DE RECUPERACIÓN (RAG) ===")
        # Consulta de prueba enfocada en encontrar reglas de negocio/campos
        consulta = "¿Cuáles son las garantías que pueden reconocer las Instituciones para la Severidad de la Pérdida?"
        
        resultados = buscar_similitud(consulta, k=2)
        
        for i, doc in enumerate(resultados):
            print(f"\n--- Resultado {i+1} ---")
            print(f"Metadatos: {doc.metadata}")
            print(f"Extracto: {doc.page_content[:300]}...")
    else:
        print(f"No se encontró el archivo de prueba: {archivo_prueba}")
