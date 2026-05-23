import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Asegurar importaciones locales
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.nlp_core.chunking import crear_chunks_markdown

class MotorVectorizacion:
    """
    Clase responsable de la creación de Embeddings y la indexación 
    de documentos en la base de datos vectorial (ChromaDB).
    """
    def __init__(self, persist_dir: str | Path = None):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key
        )
        
        self.persist_dir = str(persist_dir) if persist_dir else str(project_root / "data" / "03_output" / "chroma_db")

    def indexar_documento_markdown(self, ruta_archivo: Path, chunker=None, origen: str = "CNBV", collection_name: str = "regulacion_disf", postprocesadores: list = None):
        """
        Toma un archivo Markdown local, lo fragmenta usando el chunker proporcionado y lo guarda en ChromaDB.
        Si no se provee chunker, usa EstrategiaChunking.ENCABEZADOS_MD por defecto.
        Adicionalmente, permite inyectar postprocesadores (Filtros) para alterar el texto (Ej. Contextual Retrieval).
        """
        from src.utils.limpieza_texto import procesar_documento
        from src.nlp_core.chunking import RegulacionChunker, EstrategiaChunking
        
        if chunker is None:
            chunker = RegulacionChunker(EstrategiaChunking.ENCABEZADOS_MD, chunk_size=500, overlap=80)
            
        print(f"Fragmentando el documento: {ruta_archivo.name} con estrategia {chunker.estrategia}...")
        
        texto = ruta_archivo.read_text(encoding="utf-8")
        texto_limpio = procesar_documento(texto, origen=origen)
        chunks = chunker.chunk(texto_limpio)
        
        print(f"   Se generaron {len(chunks)} chunks originales.")
        
        if postprocesadores:
            print(f"   Aplicando {len(postprocesadores)} post-procesadores...")
            for procesador in postprocesadores:
                chunks = procesador.procesar(chunks, texto_limpio)
        
        print(f"   Chunks finales a vectorizar: {len(chunks)}")
        
        # Inyectar metadata y generar IDs deterministas para evitar duplicados
        ids = []
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = ruta_archivo.name
            ids.append(f"{ruta_archivo.stem}_{chunker.estrategia}_chunk_{i}")
            
        print("Indexando en ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            ids=ids,
            persist_directory=self.persist_dir,
            collection_name=collection_name
        )
        
        print(f"¡Documento indexado con éxito en {self.persist_dir}!")
        return vectorstore

    def indexar_dataset_unificado(self, df_textos: pd.DataFrame, collection_name: str = "regulacion_formularios_disf"):
        """
        Toma un DataFrame pandas unificado (ej. con forms, cats, y reglas) y lo convierte en documentos vectoriales.
        """
        documentos_langchain = []
        ids = []
        
        for _, row in df_textos.iterrows():
            doc_id = str(row.get("id"))
            ids.append(doc_id)
            documentos_langchain.append(
                Document(
                    page_content=row["texto"],
                    metadata={
                        "id": doc_id,
                        "tipo_documento": row.get("tipo_documento", ""),
                        "documento": row.get("documento", ""),
                        "seccion": row.get("seccion", ""),
                        "catalogo": row.get("catalogo", ""),
                        "n_palabras": row.get("n_palabras", 0)
                    }
                )
            )

        print("Indexando dataset masivo en ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=documentos_langchain,
            embedding=self.embeddings,
            ids=ids, # Usa IDs deterministas para sobreescribir y evitar duplicados
            persist_directory=self.persist_dir,
            collection_name=collection_name
        )

        print(f'"Base vectorial creada y almacenada en" {self.persist_dir}')
        return vectorstore

# =====================================================================
# FUNCIONES  
# =====================================================================

def obtener_embeddings():
    motor = MotorVectorizacion()
    return motor.embeddings

def indexar_documento(ruta_archivo: Path, chunker=None, origen: str = "CNBV", collection_name: str = "regulacion_disf", postprocesadores=None):
    motor = MotorVectorizacion()
    return motor.indexar_documento_markdown(ruta_archivo, chunker=chunker, origen=origen, collection_name=collection_name, postprocesadores=postprocesadores)

def indexar_documentos_formularios(df_textos: pd.DataFrame, collection_name: str = "regulacion_formularios_disf"):
    motor = MotorVectorizacion()
    return motor.indexar_dataset_unificado(df_textos, collection_name)

# Las funciones de búsqueda fueron movidas a retrieval.py, 
# pero las mantenemos aquí como puente para no romper los imports existentes.
from src.nlp_core.retrieval import MotorBusqueda

def buscar_similitud_chroma(query: str, vectorstore, k: int = 5):
    motor = MotorBusqueda()
    motor.vectorstore = vectorstore # Override con el vectorstore pasado
    return motor.buscar_similitud_tabular(query, k)

def buscar_similitud(query: str, collection_name: str = "regulacion_disf", k: int = 3):
    motor = MotorBusqueda(collection_name=collection_name)
    return motor.buscar_similitud(query, k)

def top_tfidf_terms(vectorizer, X, top_n=25):
    return MotorBusqueda.top_tfidf_terms(vectorizer, X, top_n)

def buscar_tfidf(query: str, vectorizer, X, df_base: pd.DataFrame, k: int = 5):
    return MotorBusqueda.buscar_tfidf(query, vectorizer, X, df_base, k)

def resumen_resultados_busqueda(nombre_consulta, resultados_tfidf, resultados_chroma):
    return MotorBusqueda.resumen_resultados_busqueda(nombre_consulta, resultados_tfidf, resultados_chroma)
