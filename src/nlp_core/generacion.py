import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH para que encuentre 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dotenv import load_dotenv

# Importar configuración de LLM centralizada
from src.nlp_core.config_llm import get_llm_client, get_llm_model_name
from src.nlp_core.prompts_registry import get_prompt

# Importamos nuestro esquema Pydantic
from src.nlp_core.schemas import RequerimientoInformacion
from src.nlp_core.retrieval import MotorBusqueda
from src.nlp_core.pipeline import PipelineRecuperacion
from langchain_core.documents import Document

_DOCS_RAW_CACHE = None
_MOTOR_CACHE = None

def get_motor_and_docs():
    global _MOTOR_CACHE, _DOCS_RAW_CACHE
    if _MOTOR_CACHE is None:
        _MOTOR_CACHE = MotorBusqueda(collection_name="regulacion_disf")
    if _DOCS_RAW_CACHE is None:
        print("Cargando documentos crudos en memoria para BM25/Híbrido...")
        data = _MOTOR_CACHE.vectorstore.get(include=['documents', 'metadatas'])
        _DOCS_RAW_CACHE = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(data['documents'], data['metadatas'])
        ]
    return _MOTOR_CACHE, _DOCS_RAW_CACHE

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Archivo de persistencia para telemetría
LOG_TELEMETRIA = Path(__file__).resolve().parent.parent.parent / "data" / "03_output" / "telemetria_llm.jsonl"

def _guardar_telemetria(telemetria: dict, estrategia: str):
    """Guarda la métrica de consumo en un archivo JSONL persistente."""
    try:
        LOG_TELEMETRIA.parent.mkdir(parents=True, exist_ok=True)
        registro = {
            "timestamp": datetime.now().isoformat(),
            "estrategia": estrategia,
            **telemetria
        }
        with open(LOG_TELEMETRIA, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro) + "\n")
    except Exception as e:
        print(f"Advertencia: No se pudo guardar la telemetría - {e}")

def extraer_full_context(texto_normativo: str) -> tuple[RequerimientoInformacion, dict]:
    """
    Toma el texto limpio de un documento normativo y utiliza OpenAI 
    para extraer la estructura tabular y los catálogos en un JSON estructurado.
    Retorna la estructura y un diccionario con telemetría (tokens, latencia).
    """
    client = get_llm_client("extraction")
    modelo_extraccion = get_llm_model_name("extraction")
    
    prompt_sistema, version_prompt, hash_prompt = get_prompt("extraccion_full_context")
    
    # Utilizamos Structured Outputs de OpenAI (disponible en pydantic >= 2.0 y openai >= 1.40)
    # garantizando que la salida cumpla perfectamente con nuestro esquema.
    t0 = time.time()
    respuesta = client.beta.chat.completions.parse(
        model=modelo_extraccion,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Texto normativo a analizar:\n\n{texto_normativo}"}
        ],
        response_format=RequerimientoInformacion,
        temperature=0.1 # Temperatura baja porque queremos extracción precisa, no creatividad
    )
    latencia = time.time() - t0
    
    telemetria = {
        "modelo": modelo_extraccion,
        "prompt_tokens": respuesta.usage.prompt_tokens if respuesta.usage else 0,
        "completion_tokens": respuesta.usage.completion_tokens if respuesta.usage else 0,
        "total_tokens": respuesta.usage.total_tokens if respuesta.usage else 0,
        "latencia_seg": round(latencia, 2),
        "prompt_version": version_prompt,
        "prompt_hash": hash_prompt
    }
    
    # Guardar en disco para el dashboard futuro
    _guardar_telemetria(telemetria, "Full Context")
    
    # La API ya nos devuelve el objeto Pydantic instanciado y validado
    return respuesta.choices[0].message.parsed, telemetria

def extraer_rag_simple(query: str, k: int = 4) -> tuple[RequerimientoInformacion, dict]:
    """
    Estrategia RAG: Utiliza ChromaDB para recuperar los chunks más relevantes 
    basado en la consulta, y le pide al LLM extraer el formulario usando SOLO ese contexto.
    Retorna la estructura y un diccionario con telemetría (tokens, latencia).
    """
    
    print(f"Recuperando contexto vectorial para: '{query}'...")
    
    t0_busqueda = time.time()
    # 1. Recuperar chunks de ChromaDB
    motor, _ = get_motor_and_docs()
    resultados = motor.buscar_similitud(query, k=k)
    latencia_busqueda = time.time() - t0_busqueda
    
    if not resultados:
        raise ValueError("No se encontró contexto en la base vectorial para esa consulta.")
        
    # 2. Agrupar chunks por documento de origen para soporte Multi-Documento (Requerimiento B4)
    docs_por_archivo = {}
    for doc in resultados:
        # Extraer el nombre del archivo o documento original (varía según cómo se indexó)
        origen = doc.metadata.get("source_file", doc.metadata.get("documento", "Normativa General"))
        if origen not in docs_por_archivo:
            docs_por_archivo[origen] = []
        docs_por_archivo[origen].append(doc)
        
    bloques_contexto = []
    for origen, docs in docs_por_archivo.items():
        bloque = f"[📜 Documento: {origen}]\n"
        for i, d in enumerate(docs, 1):
            bloque += f" - Fragmento {i}: {d.page_content.strip()}\n"
        bloques_contexto.append(bloque)
        
    contexto_recuperado = "\n\n".join(bloques_contexto)
    
    # 3. Llamar al LLM con este contexto limitado
    client = get_llm_client("extraction")
    modelo_extraccion = get_llm_model_name("extraction")
    
    prompt_sistema, version_prompt, hash_prompt = get_prompt("extraccion_rag")
    
    print(f"Enviando contexto ({len(resultados)} chunks) al LLM...")
    
    t0_llm = time.time()
    respuesta = client.beta.chat.completions.parse(
        model=modelo_extraccion, 
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Consulta del usuario: {query}\n\nFragmentos Recuperados:\n\n{contexto_recuperado}"}
        ],
        response_format=RequerimientoInformacion,
        temperature=0.1
    )
    latencia_llm = time.time() - t0_llm
    
    telemetria = {
        "modelo": modelo_extraccion,
        "prompt_tokens": respuesta.usage.prompt_tokens if respuesta.usage else 0,
        "completion_tokens": respuesta.usage.completion_tokens if respuesta.usage else 0,
        "total_tokens": respuesta.usage.total_tokens if respuesta.usage else 0,
        "latencia_busqueda_seg": round(latencia_busqueda, 2),
        "latencia_llm_seg": round(latencia_llm, 2),
        "latencia_total_seg": round(latencia_busqueda + latencia_llm, 2),
        "prompt_version": version_prompt,
        "prompt_hash": hash_prompt
    }
    
    # Guardar en disco para el dashboard futuro
    _guardar_telemetria(telemetria, "RAG Simple")
    
    return respuesta.choices[0].message.parsed, telemetria

def responder_rag_qa(query: str, k: int = 4, base_retriever: str = "embeddings", query_expansion: str = "none", post_processing: str = "none") -> tuple[str, dict, list]:
    """
    Estrategia RAG Conversacional interactiva usando el Pipeline Modular.
    Retorna el texto markdown, un diccionario con telemetría y la lista de chunks recuperados.
    """
    
    print(f"Recuperando contexto (Retriever: {base_retriever}, Expansión: {query_expansion}, Post: {post_processing}) para QA: '{query}'...")
    
    t0_busqueda = time.time()
    
    motor, docs_raw = get_motor_and_docs()
    pipeline = PipelineRecuperacion(
        motor=motor,
        documentos_raw=docs_raw,
        base_retriever=base_retriever,
        query_expansion=None if query_expansion == "none" else query_expansion,
        post_processing=None if post_processing == "none" else post_processing
    )
    
    resultados = pipeline.invoke(query, k=k)
    latencia_busqueda = time.time() - t0_busqueda
    
    if not resultados:
        raise ValueError("No se encontró contexto para esa consulta usando la estrategia seleccionada.")
        
    # 2. Agrupar chunks por documento de origen para soporte Multi-Documento (Requerimiento B4)
    docs_por_archivo = {}
    for doc in resultados:
        # Extraer el nombre del archivo o documento original (varía según cómo se indexó)
        origen = doc.metadata.get("source_file", doc.metadata.get("documento", "Normativa General"))
        if origen not in docs_por_archivo:
            docs_por_archivo[origen] = []
        docs_por_archivo[origen].append(doc)
        
    bloques_contexto = []
    for origen, docs in docs_por_archivo.items():
        bloque = f"[📜 Documento: {origen}]\n"
        for i, d in enumerate(docs, 1):
            bloque += f" - Fragmento {i}: {d.page_content.strip()}\n"
        bloques_contexto.append(bloque)
        
    contexto_recuperado = "\n\n".join(bloques_contexto)
    
    # 3. Llamar al LLM con este contexto limitado (Uso de chat.completions.create normal)
    client = get_llm_client("qa")
    modelo_qa = get_llm_model_name("qa")
    
    prompt_sistema, version_prompt, hash_prompt = get_prompt("qa_rag")
    
    print(f"Enviando contexto ({len(resultados)} chunks) al LLM para respuesta de QA...")
    
    t0_llm = time.time()
    respuesta = client.chat.completions.create(
        model=modelo_qa,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Consulta del usuario: {query}\n\nFragmentos Recuperados:\n\n{contexto_recuperado}"}
        ],
        temperature=0.3
    )
    latencia_llm = time.time() - t0_llm
    
    texto_respuesta = respuesta.choices[0].message.content
    
    telemetria = {
        "modelo": modelo_qa,
        "prompt_tokens": respuesta.usage.prompt_tokens if respuesta.usage else 0,
        "completion_tokens": respuesta.usage.completion_tokens if respuesta.usage else 0,
        "total_tokens": respuesta.usage.total_tokens if respuesta.usage else 0,
        "latencia_busqueda_seg": round(latencia_busqueda, 2),
        "latencia_llm_seg": round(latencia_llm, 2),
        "latencia_total_seg": round(latencia_busqueda + latencia_llm, 2),
        "prompt_version": version_prompt,
        "prompt_hash": hash_prompt
    }
    
    # Guardar en disco para el dashboard
    _guardar_telemetria(telemetria, "RAG Conversacional QA")
    
    # Serializar los chunks para el frontend
    chunks_recuperados = [
        {"metadata": doc.metadata, "content": doc.page_content} 
        for doc in resultados
    ]
    
    return texto_respuesta, telemetria, chunks_recuperados

# --- Prueba rápida ---
if __name__ == "__main__":
    # Texto ficticio de prueba (muy sencillo para no gastar muchos tokens)
    texto_prueba = """
    Artículo 1. Las instituciones de crédito deberán enviar mensualmente un reporte de sus créditos comerciales a la DISF.
    Dicho reporte, que denominaremos "Formulario de Créditos Comerciales Mensual", deberá contener:
    1. Identificador del Crédito: Debe ser Alfanumérico con máximo 15 caracteres.
    2. Moneda del crédito: Es obligatorio enviar la clave de moneda. Las opciones válidas son 'MXN' para Pesos Mexicanos y 'USD' para Dólares Estadounidenses.
    3. Tasa de Interés: Debe ser un valor Numérico sin límite. Ojo, esta tasa no puede ser negativa en ningún caso.
    """
    
    print("\n=======================================================")
    print("Iniciando prueba con Estrategia 1: FULL CONTEXT...")
    try:
        resultado_fc, telemetria_fc = extraer_full_context(texto_prueba)
        print("¡Extracción Full Context exitosa!")
        print(f"📊 Formulario propuesto: {resultado_fc.nombre_formulario}")
        print(f"⏱️  Telemetría: {telemetria_fc}")
    except Exception as e:
        print(f"❌ Error en Full Context: {e}")

    print("\n=======================================================")
    print("Iniciando prueba con Estrategia 2: RAG SIMPLE...")
    try:
        query = "¿Cuáles son las metodologías y cálculos para la Severidad de la Pérdida en el Apartado E?"
        resultado_rag, telemetria_rag = extraer_rag_simple(query, k=3)
        print("\n¡Extracción RAG exitosa!")
        print(f"📊 Formulario propuesto: {resultado_rag.nombre_formulario}")
        print(f"⏱️  Telemetría: {telemetria_rag}")
        
        print("\n📝 Campos identificados por el RAG:")
        for campo in resultado_rag.campos_formulario:
            print(f" - {campo.nombre_campo} ({campo.tipo_dato}): {campo.descripcion_funcional}")
            if campo.formula_calculo:
                print(f"   [Fórmula: {campo.formula_calculo}]")
            if campo.es_catalogo:
                print(f"   [Catálogo: {campo.nombre_catalogo_vinculado}]")
                
        if resultado_rag.ambiguedades_detectadas:
            print("\n⚠️ Ambigüedades detectadas:")
            for amb in resultado_rag.ambiguedades_detectadas:
                print(f" - {amb}")
    except ValueError as ve:
        print(f"⚠️ Atención: {ve}")
    except Exception as e:
        print(f"❌ Error durante la ejecución del RAG: {e}")
