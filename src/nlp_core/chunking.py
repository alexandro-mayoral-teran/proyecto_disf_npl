import os
import sys
import re
import time
from pathlib import Path
from enum import Enum
from typing import List
from dotenv import load_dotenv

# Asegurar importaciones locales
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.utils.limpieza_texto import procesar_documento

class EstrategiaChunking(str, Enum):
    PARRAFO = "parrafo"
    FIJO_OVERLAP = "fijo_overlap"
    ESTRUCTURAL = "estructural"
    ENCABEZADOS_MD = "encabezados_md"

class RegulacionChunker:
    """
    Clase principal para manejar la fragmentación (chunking) de documentos regulatorios.
    Implementa el patrón Strategy: permite instanciar un chunker con una estrategia 
    específica y parámetros personalizables.
    """
    def __init__(self, estrategia: EstrategiaChunking = EstrategiaChunking.ENCABEZADOS_MD, **kwargs):
        self.estrategia = estrategia
        self.kwargs = kwargs

    def chunk(self, texto: str) -> List[Document]:
        """
        Aplica la estrategia de chunking seleccionada al texto de entrada.
        Siempre devuelve una lista de objetos Document de LangChain para mantener homogeneidad.
        """
        if self.estrategia == EstrategiaChunking.PARRAFO:
            return self._chunking_por_parrafo(texto)
        elif self.estrategia == EstrategiaChunking.FIJO_OVERLAP:
            return self._chunking_fijo_overlap(texto)
        elif self.estrategia == EstrategiaChunking.ESTRUCTURAL:
            return self._chunking_estructural(texto)
        elif self.estrategia == EstrategiaChunking.ENCABEZADOS_MD:
            return self._chunking_encabezados_md(texto)
        else:
            raise ValueError(f"Estrategia desconocida: {self.estrategia}")

    def _chunking_por_parrafo(self, texto: str) -> List[Document]:
        """
        ##### Chunks por párrafo
        # Corta el documento por dobles saltos de línea. Desecha párrafos con menos de 'min_palabras'.
        # Pros: Limpieza rápida.
        # Contras: Riesgo de omitir reglas críticas si están en formatos muy cortos (ej. viñetas).
        """
        min_palabras = self.kwargs.get("min_palabras", 20)
        chunks = []

        for bloque in re.split(r"\n\s*\n", texto):
            bloque = bloque.strip()
            n_palabras = len(re.findall(r"\b\w+\b", bloque))

            if n_palabras >= min_palabras:
                chunks.append(Document(page_content=bloque, metadata={"estrategia": "parrafo"}))

        return chunks

    def _chunking_fijo_overlap(self, texto: str) -> List[Document]:
        """
        ##### Chunks fijos con overlap
        # Estrategia recomendada para vectorización: asegura homogeneidad de tamaño.
        # Genera fragmentos superpuestos para no perder contexto al cortar entre oraciones.
        """
        chunk_size = self.kwargs.get("chunk_size", 300)
        overlap = self.kwargs.get("overlap", 50)
        palabras = re.findall(r"\b\w+\b", texto)
        chunks = []

        start = 0
        while start < len(palabras):
            end = start + chunk_size
            chunk_texto = " ".join(palabras[start:end])
            chunks.append(Document(page_content=chunk_texto, metadata={"estrategia": "fijo_overlap"}))
            start += chunk_size - overlap

        return chunks

    def _chunking_estructural(self, texto: str) -> List[Document]:
        """
        ##### Chunking estructural por títulos/artículos
        # Utiliza expresiones regulares para fragmentar cada vez que aparece una palabra clave regulatoria.
        # Pros: Mantiene la estructura legal exacta.
        # Contras: Puede generar chunks gigantes si un artículo tiene muchas páginas, lo que desestabiliza el RAG.
        """
        patron = r"(?=(?:Artículo\s+\d+|TÍTULO\s+[A-ZÁÉÍÓÚÑ]+|CAPÍTULO\s+[A-ZÁÉÍÓÚÑ]+|Apartado\s+[A-Z]))"
        partes = re.split(patron, texto)

        chunks = []
        for parte in partes:
            parte = parte.strip()
            if len(parte.split()) >= 20:
                chunks.append(Document(page_content=parte, metadata={"estrategia": "estructural"}))

        return chunks

    def _normalizar_headers_regulatorios(self, texto: str) -> str:
        texto = re.sub(r'^(TÍTULO\s+[A-ZÁÉÍÓÚÑ]+.*)$', r'# \1', texto, flags=re.MULTILINE)
        texto = re.sub(r'^(Título\s+[A-ZÁÉÍÓÚÑ]+.*)$', r'# \1', texto, flags=re.MULTILINE)
        texto = re.sub(r'^(ANEXO\s+[A-ZÁÉÍÓÚÑ]+.*)$', r'# \1', texto, flags=re.MULTILINE)
        texto = re.sub(r'^(Anexo\s+[A-ZÁÉÍÓÚÑ]+.*)$', r'# \1', texto, flags=re.MULTILINE)
        texto = re.sub(r'^(CAPÍTULO\s+[A-ZÁÉÍÓÚÑ]+.*)$', r'## \1', texto, flags=re.MULTILINE)
        texto = re.sub(r'^(Capítulo\s+[A-ZÁÉÍÓÚÑ]+.*)$', r'## \1', texto, flags=re.MULTILINE)
        texto = re.sub(r'^(ARTÍCULO\s+\d+.*)$', r'### \1', texto, flags=re.MULTILINE)
        texto = re.sub(r'^(Artículo\s+\d+.*)$', r'### \1', texto, flags=re.MULTILINE)
        return texto

    def _chunking_encabezados_md(self, texto: str) -> List[Document]:
        """
        ##### Chunking basado en encabezados Markdown
        # Lee un texto, normaliza títulos, capítulos y artículos a formato Markdown,
        # y fragmenta respetando esta jerarquía. Usa RecursiveCharacterTextSplitter 
        # como capa secundaria para trozar párrafos inmensos.
        # Pros: Preserva los metadatos jerárquicos. Ideal para RAG avanzado.
        # Contras: Requiere mayor configuración y procesamiento previo.
        """
        chunk_size = self.kwargs.get("chunk_size", 500)
        overlap = self.kwargs.get("overlap", 80)
        
        texto = self._normalizar_headers_regulatorios(texto)

        headers_to_split_on = [
            ("#", "Header 1"),      # Ej: Títulos / Anexos
            ("##", "Header 2"),     # Ej: Capítulos
            ("###", "Header 3"),    # Ej: Secciones
            ("####", "Header 4"),   # Ej: Artículos
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False 
        )

        documentos = markdown_splitter.split_text(texto)
        
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

        chunks_finales = []

        for doc in documentos:
            subchunks = recursive_splitter.split_text(doc.page_content)

            for chunk_text in subchunks:
                metadata = doc.metadata.copy()
                metadata["estrategia"] = "encabezados_md"
                chunks_finales.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
        
        return chunks_finales

class PostProcesadorChunk:
    """Clase base para todos los post-procesadores de chunks (Filtros de Pipeline)."""
    def procesar(self, chunks: List[Document], texto_completo: str = "") -> List[Document]:
        raise NotImplementedError("Debe implementarse en la subclase")

class InyectorMetadatos(PostProcesadorChunk):
    """
    Inyecta los metadatos jerárquicos extraídos por el MarkdownHeaderTextSplitter
    directamente al inicio del page_content de cada chunk.
    """
    def procesar(self, chunks: List[Document], texto_completo: str = "") -> List[Document]:
        chunks_procesados = []
        for doc in chunks:
            # Extraer headers de los metadatos (ej. Header 1, Header 2...)
            headers = []
            for key, value in doc.metadata.items():
                if "Header" in key:
                    headers.append(f"{value}")
            
            if headers:
                contexto_str = " | ".join(headers)
                nuevo_contenido = f"[Contexto Estructural: {contexto_str}]\n\n{doc.page_content}"
            else:
                nuevo_contenido = doc.page_content
                
            chunks_procesados.append(
                Document(page_content=nuevo_contenido, metadata=doc.metadata.copy())
            )
            
        return chunks_procesados

class ContextualizadorLLM(PostProcesadorChunk):
    """
    Implementa la técnica de 'Contextual Retrieval'.
    Para cada chunk, llama a un LLM ligero (gpt-4o-mini) para que redacte
    una o dos oraciones de contexto, y las antepone al contenido.
    """
    def __init__(self, max_retries: int = 3):
        from src.nlp_core.config_llm import get_langchain_chat
        
        self.llm = get_langchain_chat(task="qa", temperature=0.0)
        self.max_retries = max_retries
        self.prompt = PromptTemplate.from_template(
            "Eres un experto regulador financiero del Banco de México.\n"
            "A continuación te presento un documento normativo completo (o una gran sección) y un fragmento (chunk) específico extraído de él.\n\n"
            "<documento_completo>\n{documento}\n</documento_completo>\n\n"
            "<fragmento>\n{chunk}\n</fragmento>\n\n"
            "Tu tarea es redactar estrictamente 1 o 2 oraciones breves que le den contexto a este fragmento basándote en el documento completo. "
            "Por ejemplo, debes mencionar a qué anexo o cartera específica (ej. tarjeta de crédito, auto, nómina) pertenece este fragmento. "
            "NO resumas el fragmento, solo explica su contexto jerárquico o el tema principal al que responde.\n"
            "Respuesta:"
        )

    def procesar(self, chunks: List[Document], texto_completo: str = "") -> List[Document]:
        chunks_procesados = []
        texto_recortado = texto_completo[:30000]
        
        print(f"Contextualizando {len(chunks)} chunks con LLM...")
        
        for i, doc in enumerate(chunks):
            print(f"  Procesando chunk {i+1}/{len(chunks)}...", end="\r")
            
            contexto_generado = ""
            intentos = 0
            while intentos < self.max_retries:
                try:
                    chain = self.prompt | self.llm
                    respuesta = chain.invoke({
                        "documento": texto_recortado,
                        "chunk": doc.page_content
                    })
                    contexto_generado = respuesta.content.strip()
                    break
                except Exception as e:
                    intentos += 1
                    time.sleep(1) # Backoff simple
                    if intentos == self.max_retries:
                        print(f"\n  [!] Error al contextualizar chunk {i}: {e}. Se omitirá el contexto LLM.")
                        contexto_generado = ""

            if contexto_generado:
                nuevo_contenido = f"[Contexto generado por IA: {contexto_generado}]\n\n{doc.page_content}"
            else:
                nuevo_contenido = doc.page_content
                
            chunks_procesados.append(
                Document(page_content=nuevo_contenido, metadata=doc.metadata.copy())
            )
            
        print("\nContextualización con LLM finalizada.")
        return chunks_procesados

# =====================================================================
# FUNCIONES DE COMPATIBILIDAD HACIA ATRÁS (Para no romper otros módulos)
# Estas funciones envuelven a la nueva clase RegulacionChunker
# =====================================================================

def chunking_por_parrafo(texto: str, min_palabras: int = 20) -> list:
    chunker = RegulacionChunker(EstrategiaChunking.PARRAFO, min_palabras=min_palabras)
    # Devolvemos strings para mantener compatibilidad con cuadernos Jupyter viejos
    return [d.page_content for d in chunker.chunk(texto)]

def chunking_fijo_overlap(texto: str, chunk_size: int = 300, overlap: int = 50) -> list:
    chunker = RegulacionChunker(EstrategiaChunking.FIJO_OVERLAP, chunk_size=chunk_size, overlap=overlap)
    return [d.page_content for d in chunker.chunk(texto)]

def chunking_estructural(texto: str) -> list:
    chunker = RegulacionChunker(EstrategiaChunking.ESTRUCTURAL)
    return [d.page_content for d in chunker.chunk(texto)]

def chunking_encabezados_md(texto: str, chunk_size: int = 500, overlap: int = 80) -> list:
    chunker = RegulacionChunker(EstrategiaChunking.ENCABEZADOS_MD, chunk_size=chunk_size, overlap=overlap)
    return chunker.chunk(texto) # Ya devolvía Documents

def crear_chunks_markdown(ruta_archivo: Path, origen: str = "CNBV") -> list:
    """
    Función de compatibilidad utilizada por vectorizacion.py.
    """
    texto = ruta_archivo.read_text(encoding="utf-8")
    texto_limpio = procesar_documento(texto, origen=origen)
    return chunking_encabezados_md(texto_limpio)

if __name__ == "__main__":
    # --- PRUEBA RÁPIDA ---
    archivo_prueba = project_root / "data" / "02_interim"/ "CUB_extracto.md"
    
    if archivo_prueba.exists():
        print(f"Fragmentando archivo: {archivo_prueba.name}...")

        texto = archivo_prueba.read_text(encoding="utf-8")
        texto_limpio = procesar_documento(texto, origen="CNBV")
        
        # Probamos el chunker Orientado a Objetos
        chunker = RegulacionChunker(EstrategiaChunking.ENCABEZADOS_MD, chunk_size=500, overlap=80)
        chunks_generados = chunker.chunk(texto_limpio)
        
        print(f"Total de chunks generados: {len(chunks_generados)}\n")
        
        if chunks_generados:
            print("="*50)
            print("--- EJEMPLO DEL PRIMER CHUNK ---")
            print(f"Metadatos: {chunks_generados[0].metadata}")
            print(f"Contenido (primeros 400 caracteres):\n{chunks_generados[0].page_content[:400]}...")
            
            print("\n" + "="*50)
            print("--- EJEMPLO DEL ÚLTIMO CHUNK ---")
            print(f"Metadatos: {chunks_generados[-1].metadata}")
            print(f"Contenido (primeros 400 caracteres):\n{chunks_generados[-1].page_content[:400]}...")
    else:
        print(f"No se encontró el archivo de prueba en {archivo_prueba}")
