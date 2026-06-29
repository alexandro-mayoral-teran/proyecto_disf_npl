import os
import sys
import time
import json
import pickle
from datetime import datetime
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH para que encuentre 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dotenv import load_dotenv

# Importar configuración de LLM centralizada
from src.nlp_core.config_llm import get_llm_client, get_llm_model_name
from src.nlp_core.prompts_registry import get_prompt

# Importamos nuestro esquema Pydantic
from src.nlp_core.schemas import RequerimientoInformacion, MetadataDocumento
from src.nlp_core.retrieval import MotorBusqueda
from src.nlp_core.pipeline import PipelineRecuperacion
from src.nlp_core.seguridad.guardrails import verificar_input_seguro
from langchain_core.documents import Document

_DOCS_RAW_CACHE = None
_MOTOR_CACHE = None
_CURRENT_DB_FOLDER = None

def get_motor_and_docs(db_folder: str = "chroma_db"):
    global _MOTOR_CACHE, _DOCS_RAW_CACHE, _CURRENT_DB_FOLDER
    if _MOTOR_CACHE is None or _CURRENT_DB_FOLDER != db_folder:
        print(f"[RELOAD] Configurando Motor de Búsqueda apuntando a: {db_folder}")
        persist_dir = Path(__file__).resolve().parent.parent.parent / "data" / "03_output" / db_folder
        _MOTOR_CACHE = MotorBusqueda(persist_dir=persist_dir, collection_name="regulacion_disf")
        _CURRENT_DB_FOLDER = db_folder
        _DOCS_RAW_CACHE = None # Forzar recarga de documentos en RAM
        
        from src.nlp_core.retrieval import limpiar_cache_retrieval
        limpiar_cache_retrieval()
        
    if _DOCS_RAW_CACHE is None:
        cache_path = Path(__file__).resolve().parent.parent.parent / "data" / "03_output" / f"docs_cache_{db_folder}.pkl"
        use_cache = os.getenv("USE_IN_MEMORY_CACHE", "true").lower() != "false"
        
        if use_cache and cache_path.exists():
            print("[LOAD] Cargando documentos desde caché binario (Pickle)...")
            with open(cache_path, "rb") as f:
                _DOCS_RAW_CACHE = pickle.load(f)
        else:
            print("[EXTRACT] Extrayendo documentos crudos desde VectorDB...")
            data = _MOTOR_CACHE.vectorstore.get(include=['documents', 'metadatas'])
            _DOCS_RAW_CACHE = [
                Document(page_content=txt, metadata=meta)
                for txt, meta in zip(data['documents'], data['metadatas'])
            ]
            if use_cache:
                print("[SAVE] Guardando documentos en caché binario...")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(_DOCS_RAW_CACHE, f)
                    
    return _MOTOR_CACHE, _DOCS_RAW_CACHE

_SEMANTIC_CACHE = None
SEMANTIC_CACHE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "03_output" / "semantic_cache.pkl"

def _load_semantic_cache():
    global _SEMANTIC_CACHE
    if _SEMANTIC_CACHE is None:
        if SEMANTIC_CACHE_FILE.exists():
            with open(SEMANTIC_CACHE_FILE, "rb") as f:
                _SEMANTIC_CACHE = pickle.load(f)
        else:
            _SEMANTIC_CACHE = []
    return _SEMANTIC_CACHE

def _check_semantic_cache(query: str, db_folder: str = "chroma_db", threshold_math: float = 0.80):
    """Verifica si la pregunta actual es semánticamente idéntica usando un Juez LLM."""
    cache_total = _load_semantic_cache()
    if not cache_total: return None
    
    # Filtrar solo por la base de datos actual
    cache = [item for item in cache_total if item.get("db_folder", "chroma_db") == db_folder]
    if not cache: return None
    
    from src.nlp_core.config_llm import get_embeddings, get_llm_client, get_llm_model_name
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    emb = get_embeddings()
    q_vec = np.array(emb.embed_query(query)).reshape(1, -1)
    
    # 1. Filtro Matemático Rápido
    cached_vecs = np.array([item["embedding"] for item in cache])
    sims = cosine_similarity(q_vec, cached_vecs)[0]
    
    best_idx = np.argmax(sims)
    mejor_similitud = sims[best_idx]
    
    print(f"[CACHE] Evaluando caché... Máxima similitud matemática encontrada: {mejor_similitud:.4f}")
    
    if mejor_similitud >= threshold_math:
        # 2. El Juez LLM
        pregunta_cache = cache[best_idx]["query"]
        print(f"[JUDGE] Similitud > {threshold_math}. Activando LLM Juez para comparar:\n - N: '{query}'\n - C: '{pregunta_cache}'")
        
        client = get_llm_client("qa")  # Usamos Ollama local
        modelo_qa = get_llm_model_name("qa")
        
        prompt_juez = f"""Eres un evaluador estricto. Revisa estas dos preguntas del usuario. Tu único objetivo es determinar si ambas preguntas están buscando EXACTAMENTE la misma información o concepto normativo, sin importar variaciones gramaticales, sinónimos o errores ortográficos. Responde ÚNICAMENTE con la palabra "SI" o la palabra "NO", sin puntos ni explicaciones.

Pregunta Nueva: {query}
Pregunta en Caché: {pregunta_cache}"""

        t0_juez = time.time()
        try:
            # Forzamos max_tokens a algo muy pequeño para latencia instantánea
            resp = client.chat.completions.create(
                model=modelo_qa,
                messages=[{"role": "user", "content": prompt_juez}],
                temperature=0.0,
                max_tokens=4
            )
            veredicto = resp.choices[0].message.content.strip().upper()
            latencia_juez = time.time() - t0_juez
            
            if "SI" in veredicto:
                print(f"[OK] LLM Juez dictaminó 'SI' en {latencia_juez:.2f}s -> CACHE HIT!")
                return cache[best_idx]
            else:
                print(f"[FAIL] LLM Juez dictaminó 'NO' en {latencia_juez:.2f}s -> CACHE MISS!")
        except Exception as e:
            print(f"[WARN] Error en LLM Juez: {e}. Cayendo a Cache Miss.")
    else:
        print(f"[FAIL] La similitud matemática ({mejor_similitud:.4f}) no superó el umbral mínimo ({threshold_math}).")
        
    return None

def _save_to_semantic_cache(query: str, respuesta: str, meta: dict, chunks: list, db_folder: str = "chroma_db"):
    """Guarda la respuesta exitosa en el caché semántico persistente."""
    use_cache = os.getenv("USE_IN_MEMORY_CACHE", "true").lower() != "false"
    if not use_cache: return
    
    cache = _load_semantic_cache()
    
    from src.nlp_core.config_llm import get_embeddings
    emb = get_embeddings()
    q_vec = emb.embed_query(query)
    
    cache.append({
        "query": query,
        "embedding": q_vec,
        "respuesta": respuesta,
        "meta": meta,
        "chunks": chunks,
        "db_folder": db_folder
    })
    
    # Guardar a disco para que sobreviva a Streamlit / Reinicios
    SEMANTIC_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SEMANTIC_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

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

def extraer_full_context(texto_normativo: str, instrucciones: str = None) -> tuple[RequerimientoInformacion, dict]:
    """
    Toma el texto limpio de un documento normativo y utiliza OpenAI 
    para extraer la estructura tabular y los catálogos en un JSON estructurado.
    Retorna la estructura y un diccionario con telemetría (tokens, latencia).
    """
    client = get_llm_client("extraction")
    modelo_extraccion = get_llm_model_name("extraction")
    
    prompt_sistema, version_prompt, hash_prompt = get_prompt("extraccion_full_context")
    
    # Inyectar instrucciones personalizadas si existen
    user_content = f"Texto normativo a analizar:\n\n{texto_normativo}"
    if instrucciones:
        user_content += f"\n\n[INSTRUCCIONES ESPECÍFICAS DEL USUARIO]:\n{instrucciones}\n"
    
    # Utilizamos Structured Outputs de OpenAI (disponible en pydantic >= 2.0 y openai >= 1.40)
    # garantizando que la salida cumpla perfectamente con nuestro esquema.
    t0 = time.time()
    respuesta = client.beta.chat.completions.parse(
        model=modelo_extraccion,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": user_content}
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

def extraer_metadatos_documento(texto_normativo: str, instrucciones: str = None) -> tuple[MetadataDocumento, dict]:
    """
    Toma los primeros 4000 caracteres de un documento normativo y utiliza OpenAI 
    para extraer la metadata (tema, confidencialidad, etc.) en un JSON estructurado.
    Retorna la estructura Pydantic y un diccionario con telemetría.
    """
    client = get_llm_client("extraction")
    modelo_extraccion = get_llm_model_name("extraction")
    
    prompt_sistema, version_prompt, hash_prompt = get_prompt("extraccion_metadatos")
    
    texto_para_analizar = texto_normativo[:4000]
    
    # Inyectar instrucciones personalizadas si existen
    user_content = f"Texto normativo a analizar:\n\n{texto_para_analizar}"
    if instrucciones:
        user_content += f"\n\n[INSTRUCCIONES ESPECÍFICAS DEL USUARIO]:\n{instrucciones}\n"
    
    t0 = time.time()
    respuesta = client.beta.chat.completions.parse(
        model=modelo_extraccion,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": user_content}
        ],
        response_format=MetadataDocumento,
        temperature=0.0
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
    
    _guardar_telemetria(telemetria, "Extraccion Metadatos")
    
    return respuesta.choices[0].message.parsed, telemetria


def extraer_rag_simple(query: str, k: int = 4, tema: str = None, textos_efimeros: list = None, solo_efimero: bool = False, db_folder: str = "chroma_db") -> tuple[RequerimientoInformacion, dict]:
    """
    Estrategia RAG: Utiliza ChromaDB para recuperar los chunks más relevantes 
    basado en la consulta, y le pide al LLM extraer el formulario usando SOLO ese contexto.
    Soporta búsqueda efímera en memoria si se envían textos_efimeros.
    Retorna la estructura y un diccionario con telemetría (tokens, latencia).
    """
    
    print(f"Recuperando contexto vectorial para: '{query}' (Tema: {tema}, DB: {db_folder})...")
    
    t0_busqueda = time.time()
    # 1. Recuperar chunks de ChromaDB
    motor, _ = get_motor_and_docs(db_folder)
    
    if textos_efimeros:
        resultados = motor.buscar_similitud_dinamica(query, k, textos_efimeros, solo_efimero, filtro_dominio=tema)
    else:
        resultados = motor.buscar_similitud(query, k=k, filtro_dominio=tema)
        
    latencia_busqueda = time.time() - t0_busqueda
    
    if not resultados:
        raise ValueError("No se encontró contexto para esa consulta usando el tema seleccionado.")
        
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

def responder_rag_qa(query: str, k: int = 4, base_retriever: str = "embeddings", query_expansion: str = "none", post_processing: str = "none", tema: str = None, textos_efimeros: list = None, solo_efimero: bool = False, db_folder: str = "chroma_db") -> tuple[str, dict, list]:
    """
    Estrategia RAG Conversacional interactiva usando el Pipeline Modular.
    Retorna el texto markdown, un diccionario con telemetría y la lista de chunks recuperados.
    """
    
    print(f"Recuperando contexto (Retriever: {base_retriever}, Tema: {tema}, DB: {db_folder}, Efímero: {bool(textos_efimeros)}) para QA: '{query}'...")
    
    t0_busqueda = time.time()
    
    motor, docs_raw = get_motor_and_docs(db_folder)
    
    if textos_efimeros:
        # Bypassear el pipeline regular si hay textos efímeros
        resultados = motor.buscar_similitud_dinamica(query, k, textos_efimeros, solo_efimero, filtro_dominio=tema)
    else:
        pipeline = PipelineRecuperacion(
            motor=motor,
            documentos_raw=docs_raw,
            base_retriever=base_retriever,
            query_expansion=None if query_expansion == "none" else query_expansion,
            post_processing=None if post_processing == "none" else post_processing
        )
        
        resultados = pipeline.invoke(query, k=k, filtro_dominio=tema)
        
    latencia_busqueda = time.time() - t0_busqueda
    
    if not resultados:
        raise ValueError("No se encontró contexto para esa consulta usando el tema seleccionado.")
        
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
def responder_rag_cascade_qa(
    query: str, 
    k: int = 13, 
    umbral_faithfulness: float = 0.8, 
    base_retriever: str = "hibrido",          # Opciones: "embeddings", "hibrido", "bm25", "tfidf", "bow"
    query_expansion: str = "none",            # Opciones: "none", "multi_query", "hyde", "ambos"
    post_processing: str = "cross_encoder",   # Opciones: "none", "cross_encoder"
    tema: str = None,
    textos_efimeros: list = None,
    solo_efimero: bool = False,
    db_folder: str = "chroma_db"
) -> tuple[str, dict, list]:
    """
    Patrón Ensamble Heterogéneo: Router/Cascade por Confianza.
    Intenta responder primero con el modelo local. Luego evalúa su 'Faithfulness'.
    Si el LLM local tiene alucinaciones o baja fidelidad (score < umbral), escala a la nube.
    """
    print(f"[*] Iniciando RAG Cascade para: '{query}'")
    
    # === INPUT GUARDRAIL ===
    is_safe, reason = verificar_input_seguro(query)
    if not is_safe:
        print(f"[!] ALERTA ROJA: Consulta maliciosa bloqueada. Motivo: {reason}")
        telemetria_bloqueo = {
            "estrategia_cascade": "Bloqueado por Seguridad",
            "latencia_total_seg": 0.5,
            "faithfulness_local_score": 0.0,
            "motivo_bloqueo": reason
        }
        return f"[LOCKED] **Petición Bloqueada**: La consulta ha sido rechazada por nuestras políticas de seguridad institucionales ({reason}).", telemetria_bloqueo, []
    
    # === SEMANTIC CACHE CHECK ===
    use_cache = os.getenv("USE_IN_MEMORY_CACHE", "true").lower() != "false"
    if use_cache:
        cached_res = _check_semantic_cache(query, db_folder)
        if cached_res:
            meta = cached_res["meta"].copy()
            meta["estrategia_cascade"] = "Semantic Cache Hit"
            meta["latencia_total_seg"] = 0.05
            return cached_res["respuesta"], meta, cached_res["chunks"]
    
    # Asegurarnos de que el primer intento es LOCAL
    estado_original_qa = os.getenv("USE_LOCAL_QA", "false")
    os.environ["USE_LOCAL_QA"] = "true"
    
    meta_local = {}
    try:
        resp_local, meta_local, chunks = responder_rag_qa(
            query, k, 
            base_retriever=base_retriever,
            query_expansion=query_expansion,
            post_processing=post_processing,
            tema=tema,
            textos_efimeros=textos_efimeros,
            solo_efimero=solo_efimero,
            db_folder=db_folder
        )
        contexto_str = "\n".join([c["content"] for c in chunks])
        
        # Evaluar Confianza (Faithfulness)
        from src.nlp_core.evals.evaluador import evaluar_faithfulness_claims
        score = evaluar_faithfulness_claims(resp_local, contexto_str)
        
        if score >= umbral_faithfulness:
            print(f"[OK] Respuesta Local aceptada (Faithfulness: {score:.2f})")
            meta_local["faithfulness_local_score"] = round(score, 2)
            meta_local["estrategia_cascade"] = "Resuelto Local"
            
            _save_to_semantic_cache(query, resp_local, meta_local, chunks, db_folder)
            
            # Restaurar variable original
            os.environ["USE_LOCAL_QA"] = estado_original_qa
            return resp_local, meta_local, chunks
        else:
            print(f"[WARN] Baja Confianza Local (Faithfulness: {score:.2f} < {umbral_faithfulness}). Redirigiendo a Nube...")
            
    except Exception as e:
        print(f"[WARN] Error en capa local, forzando fallback a nube: {e}")
        score = 0.0
        chunks = []
        
    # === FALLBACK A LA NUBE ===
    os.environ["USE_LOCAL_QA"] = "false"
    
    try:
        t0_nube = time.time()
        resp_nube, meta_nube, chunks_nube = responder_rag_qa(
            query, k, 
            base_retriever=base_retriever,
            query_expansion=query_expansion,
            post_processing=post_processing,
            tema=tema,
            textos_efimeros=textos_efimeros,
            solo_efimero=solo_efimero,
            db_folder=db_folder
        )
        
        meta_nube["faithfulness_local_score"] = round(score, 2)
        meta_nube["estrategia_cascade"] = "Escalado a Nube"
        meta_nube["latencia_cascade_total"] = round(time.time() - t0_nube + meta_local.get("latencia_total_seg", 0), 2)
        
        _save_to_semantic_cache(query, resp_nube, meta_nube, chunks_nube, db_folder)
        
        return resp_nube, meta_nube, chunks_nube
    finally:
        # Restaurar configuración original SIEMPRE
        os.environ["USE_LOCAL_QA"] = estado_original_qa

def extraer_rag_cascade(query: str, k: int = 4, umbral_faithfulness: float = 0.8, tema: str = None, textos_efimeros: list = None, solo_efimero: bool = False, db_folder: str = "chroma_db") -> tuple[RequerimientoInformacion, dict]:
    """Wrapper para Cascade en caso de usarse para formularios estructurados."""
    # Como los formularios son Pydantic, la evaluación de Faithfulness sobre JSON es más estricta.
    # Por ahora simplemente envolvemos la función para cumplir el API del walkthrough.
    print(f"[*] Iniciando Extracción RAG Cascade para: '{query}'")
    
    # === INPUT GUARDRAIL ===
    is_safe, reason = verificar_input_seguro(query)
    if not is_safe:
        print(f"[!] ALERTA ROJA: Consulta maliciosa bloqueada. Motivo: {reason}")
        raise ValueError(f"Petición Bloqueada: {reason}")
        
    estado_original = os.getenv("USE_LOCAL_EXTRACTION", "false")
    os.environ["USE_LOCAL_EXTRACTION"] = "true"
    
    try:
        resultado, meta = extraer_rag_simple(query, k, tema=tema, textos_efimeros=textos_efimeros, solo_efimero=solo_efimero, db_folder=db_folder)
        meta["faithfulness_local_score"] = 1.0 # Asumimos 1.0 temporalmente por falta de contexto en texto plano
        meta["estrategia_cascade"] = "Resuelto Local (Extracción)"
        os.environ["USE_LOCAL_EXTRACTION"] = estado_original
        return resultado, meta
    except Exception as e:
        print(f"[WARN] Falla en extracción local: {e}. Redirigiendo a Nube...")
        os.environ["USE_LOCAL_EXTRACTION"] = "false"
        try:
            resultado_nube, meta_nube = extraer_rag_simple(query, k, tema=tema, textos_efimeros=textos_efimeros, solo_efimero=solo_efimero, db_folder=db_folder)
            meta_nube["estrategia_cascade"] = "Escalado a Nube (Extracción)"
            return resultado_nube, meta_nube
        finally:
            os.environ["USE_LOCAL_EXTRACTION"] = estado_original


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
        print(f"[DATA] Formulario propuesto: {resultado_fc.nombre_formulario}")
        print(f"[TIME]  Telemetría: {telemetria_fc}")
    except Exception as e:
        print(f"[FAIL] Error en Full Context: {e}")

    print("\n=======================================================")
    print("Iniciando prueba con Estrategia 2: RAG SIMPLE...")
    try:
        query = "¿Cuáles son las metodologías y cálculos para la Severidad de la Pérdida en el Apartado E?"
        resultado_rag, telemetria_rag = extraer_rag_simple(query, k=3)
        print("\n¡Extracción RAG exitosa!")
        print(f"[DATA] Formulario propuesto: {resultado_rag.nombre_formulario}")
        print(f"[TIME]  Telemetría: {telemetria_rag}")
        
        print("\n[NOTE] Campos identificados por el RAG:")
        for campo in resultado_rag.campos_formulario:
            print(f" - {campo.nombre_campo} ({campo.tipo_dato}): {campo.descripcion_funcional}")
            if campo.formula_calculo:
                print(f"   [Fórmula: {campo.formula_calculo}]")
            if campo.es_catalogo:
                print(f"   [Catálogo: {campo.nombre_catalogo_vinculado}]")
                
        if resultado_rag.ambiguedades_detectadas:
            print("\n[WARN] Ambigüedades detectadas:")
            for amb in resultado_rag.ambiguedades_detectadas:
                print(f" - {amb}")
    except ValueError as ve:
        print(f"[WARN] Atención: {ve}")
    except Exception as e:
        print(f"[FAIL] Error durante la ejecución del RAG: {e}")
