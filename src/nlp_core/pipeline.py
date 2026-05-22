class PipelineRecuperacion:
    """
    RAG Modular Pipeline para experimentación.
    Permite encadenar dinámicamente:
    1. Expansión de Consultas (Pre-procesamiento)
    2. Recuperación Base (BoW, TF-IDF, BM25, Embeddings, Híbrido)
    3. Reordenamiento / Reranking (Post-procesamiento)
    """
    def __init__(self, motor, documentos_raw, base_retriever="embeddings", query_expansion=None, post_processing=None, hybrid_weights=[0.5, 0.5]):
        self.motor = motor
        self.documentos_raw = documentos_raw
        
        self.base_retriever = base_retriever
        self.query_expansion = query_expansion
        self.post_processing = post_processing
        self.hybrid_weights = hybrid_weights

    def invoke(self, query: str, k: int = 5):
        # 1. Expansión de Consultas
        if self.query_expansion == "multi_query":
            queries = self._generar_multi_queries(query)
        else:
            queries = [query]
        
        # 2. Base Retrieval
        # Tomamos un top K más grande si hay reranking, para que el reranker tenga material.
        top_k_base = 20 if self.post_processing == "cross_encoder" else k
        
        # Helper interno para extraer solo 'docs' del resultado (ignorando DataFrame)
        def _get_docs(resultado):
            if isinstance(resultado, tuple) and len(resultado) == 2:
                return resultado[1]
            return resultado

        docs_totales = []
        doc_ids_vistos = set()
        
        for q in queries:
            if self.base_retriever == "bow":
                res = self.motor.buscar_bow(q, self.documentos_raw, k=top_k_base)
                docs = _get_docs(res)
            elif self.base_retriever == "tfidf":
                res = self.motor.buscar_tfidf(q, self.documentos_raw, k=top_k_base)
                docs = _get_docs(res)
            elif self.base_retriever == "bm25":
                res = self.motor.buscar_bm25(q, self.documentos_raw, k=top_k_base)
                docs = _get_docs(res)
            elif self.base_retriever == "embeddings":
                res = self.motor.buscar_similitud(q, k=top_k_base)
                docs = _get_docs(res)
            elif self.base_retriever == "hibrido":
                res = self.motor.buscar_hibrido(q, self.documentos_raw, k=top_k_base, weights=self.hybrid_weights)
                docs = _get_docs(res)
            else:
                raise ValueError(f"Retriever base no soportado: {self.base_retriever}")
                
            for d in docs:
                doc_id = d.metadata.get("id")
                if doc_id not in doc_ids_vistos:
                    doc_ids_vistos.add(doc_id)
                    docs_totales.append(d)

        # 3. Post-procesamiento (Reranking)
        if self.post_processing == "cross_encoder":
            # Usar la query original para el reranking preciso contra los resultados acumulados
            df, docs_finales = self._ejecutar_cross_encoder(query, docs_totales, k)
        else:
            # Si no hay post_processing, truncamos a k si habíamos pedido más
            docs_finales = docs_totales[:k]

        return docs_finales

    def _generar_multi_queries(self, query: str) -> list[str]:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        import warnings
        
        # Ignorar warnings de pydantic en Langchain
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
        
        print(f"Generando variantes (Multi-Query) para: '{query}'...")
        llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=self.motor.api_key)
        
        prompt = PromptTemplate(
            input_variables=["pregunta"],
            template="""Eres un asistente experto en sistemas de búsqueda corporativos. 
Tu tarea es generar 3 versiones diferentes de la pregunta dada para recuperar documentos relevantes de una base de datos de normativas financieras.
Al generar múltiples perspectivas sobre la pregunta del usuario, tu objetivo es ayudar al usuario a superar algunas de las limitaciones de la búsqueda por coincidencia exacta.
Proporciona estas preguntas alternativas separadas por saltos de línea. NO enumeres las preguntas ni agregues ningún otro texto, explicaciones, prefijos ni viñetas. Solo escribe las preguntas.
Pregunta original: {pregunta}"""
        )
        
        cadena = prompt | llm
        try:
            respuesta = cadena.invoke({"pregunta": query})
            # Limpiar contenido
            variantes = [v.strip().lstrip("-").lstrip("1234567890.). ") for v in respuesta.content.split('\n') if v.strip()]
            queries = [query] + variantes
            # Limitar a 4 total en caso de que el LLM alucine
            return queries[:4]
        except Exception as e:
            print(f"[ERROR] Falló Multi-Query: {e}. Retornando query original.")
            return [query]



    def _ejecutar_cross_encoder(self, query: str, docs_candidatos: list, k: int):
        from sentence_transformers.cross_encoder import CrossEncoder
        import numpy as np
        import pandas as pd
        
        modelo_ce = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        encoder = CrossEncoder(modelo_ce)
        
        pares = [[query, doc.page_content] for doc in docs_candidatos]
        scores = encoder.predict(pares)
        
        indices_ordenados = np.argsort(scores)[::-1][:k]
        docs_ordenados = [docs_candidatos[i] for i in indices_ordenados]
        
        filas = []
        for rank, (doc, score) in enumerate(zip(docs_ordenados, scores[indices_ordenados]), start=1):
            filas.append({
                "rank": rank,
                "score_distancia": float(score),
                "id": doc.metadata.get("id"),
                "tipo_documento": doc.metadata.get("tipo_documento"),
                "documento": doc.metadata.get("documento"),
                "texto": doc.page_content[:500]
            })
        return pd.DataFrame(filas), docs_ordenados
