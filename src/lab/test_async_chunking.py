import os
import sys
import time
from pathlib import Path

# Configurar path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.nlp_core.chunking import RegulacionChunker, EstrategiaChunking, ContextualizadorLLM
from langchain_chroma import Chroma
from src.nlp_core.config_llm import get_embeddings

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prueba de Contextual Retrieval Asíncrono")
    parser.add_argument("--compare", action="store_true", help="Ejecutar también la versión secuencial para comparar tiempos")
    args = parser.parse_args()
    
    from dotenv import load_dotenv
    load_dotenv()
    print("=== Prueba de Ingesta Asíncrona (Contextual Retrieval) ===")
    
    # 1. Leer el documento real
    file_path = project_root / "data" / "02_interim" / "markdown" / "CUB_237-278.md"
    print(f"\n1. Leyendo documento de prueba: {file_path.name}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        texto_base = f.read()
    
    print(f"-> Documento leído. Tamaño: {len(texto_base)} caracteres.")
        
    # 2. Chunking normal
    print("\n2. Ejecutando chunking fijo_overlap...")
    chunker = RegulacionChunker(EstrategiaChunking.FIJO_OVERLAP, chunk_size=300, overlap=50)
    chunks = chunker.chunk(texto_base)
    
    # Inyectar metadatos manualmente para que la API y el Frontend sepan de dónde vienen
    for chunk in chunks:
        chunk.metadata["documento"] = "CUB_237-278.md"
        chunk.metadata["tema"] = "regulacion_general"
        
    print(f"-> Se generaron {len(chunks)} chunks.")
    
    # 3. Contextual Retrieval (Versión Asíncrona Rápida)
    print("\n3. Iniciando Contextualizador LLM Asíncrono...")
    start_time_async = time.time()
    
    contextualizador = ContextualizadorLLM(max_retries=3, max_concurrent=20)
    chunks_contextualizados = contextualizador.procesar(chunks, texto_base, asincrono=True)
    
    elapsed_async = time.time() - start_time_async
    print(f"-> Contextualización ASÍNCRONA completada en {elapsed_async:.2f} segundos.")
    
    if args.compare:
        # 4. Contextual Retrieval (Versión Síncrona/Respaldo)
        print("\n4. Iniciando Contextualizador LLM de Respaldo (Secuencial)...")
        start_time_sync = time.time()
        
        chunks_contextualizados_lentos = contextualizador.procesar(chunks, texto_base, asincrono=False)
        
        elapsed_sync = time.time() - start_time_sync
        print(f"-> Contextualización SECUENCIAL completada en {elapsed_sync:.2f} segundos.")
        
        # Comparación de tiempos
        print(f"\nRESULTADOS: La versión asíncrona fue {elapsed_sync / elapsed_async:.1f} veces más rápida.")
    else:
        print("\n4. [!] Versión secuencial de respaldo omitida. (Usa --compare si deseas ejecutarla y comparar tiempos).")
    
    # 5. Guardar en ChromaDB (directorio de prueba)
    output_dir = project_root / "data" / "03_output" / "chroma_db_test_async"
    print(f"\n5. Guardando en ChromaDB de prueba: {output_dir}")
    
    # Limpiar BD de prueba anterior si existe
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks_contextualizados,
        embedding=embeddings,
        persist_directory=str(output_dir),
        collection_name="regulacion_disf"
    )
    
    print("✅ Ingesta finalizada correctamente.")

if __name__ == "__main__":
    # python src/lab/test_async_chunking.py
    # python src/lab/test_async_chunking.py --compare
    main()
