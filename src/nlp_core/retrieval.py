import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Asegurar importaciones locales
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class MotorBusqueda:
    """
    Clase responsable de la recuperación de información (Retrieval) desde la base de datos
    vectorial y otros métodos estadísticos (TF-IDF).
    """
    def __init__(self, persist_dir: str | Path = None, collection_name: str = "regulacion_disf"):
        from src.nlp_core.config_llm import get_embeddings
        self.embeddings = get_embeddings()
        
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        
        self.persist_dir = str(persist_dir) if persist_dir else str(project_root / "data" / "03_output" / "chroma_db")
        
        # Conexión a la colección específica
        self.vectorstore = Chroma(
            persist_directory=self.persist_dir, 
            embedding_function=self.embeddings,
            collection_name=collection_name
        )

    def buscar_similitud(self, query: str, k: int = 3) -> list:
        """
        Búsqueda semántica básica. Devuelve una lista de Documentos LangChain.
        """
        print(f"Buscando: '{query}'...")
        return self.vectorstore.similarity_search(query, k=k)

    def buscar_similitud_tabular(self, query: str, k: int = 5) -> tuple[pd.DataFrame, list]:
        """
        Ejecuta búsqueda semántica en ChromaDB y devuelve los resultados formateados en 
        un DataFrame de Pandas y la lista raw de documentos.
        """
        resultados = self.vectorstore.similarity_search_with_score(query, k=k)
        filas = []

        for rank, (doc, score) in enumerate(resultados, start=1):
            filas.append({
                "rank": rank,
                "score_distancia": score,
                "id": doc.metadata.get("id"),
                "tipo_documento": doc.metadata.get("tipo_documento"),
                "documento": doc.metadata.get("documento"),
                "seccion": doc.metadata.get("seccion"),
                "catalogo": doc.metadata.get("catalogo"),
                "n_palabras": doc.metadata.get("n_palabras"),
                "texto": doc.page_content[:500]
            })

        return pd.DataFrame(filas), resultados

    def buscar_hibrido(self, query: str, documentos_bm25: list, k: int = 5, weights: list = [0.5, 0.5]) -> tuple[pd.DataFrame, list]:
        """
        Ejecuta Búsqueda Híbrida: combina Búsqueda Léxica (BM25) y Semántica (ChromaDB) 
        usando Reciprocal Rank Fusion (RRF).
        Requiere la lista de documentos en memoria para inicializar el índice BM25.
        """
        from langchain_community.retrievers import BM25Retriever

        print(f"Ejecutando Búsqueda Híbrida para: '{query}'...")

        if not documentos_bm25:
            raise ValueError("Búsqueda Híbrida requiere 'documentos_bm25' para inicializar el BM25 local.")

        # 1. Búsqueda Léxica
        bm25_retriever = BM25Retriever.from_documents(documentos_bm25)
        bm25_retriever.k = k
        res_bm25 = bm25_retriever.invoke(query)
        
        # 2. Búsqueda Semántica
        res_chroma = self.vectorstore.similarity_search(query, k=k)
        
        # 3. Reciprocal Rank Fusion (RRF) manual
        rrf_k = 60
        scores = {}
        docs_map = {}
        
        for idx, doc in enumerate(res_bm25, start=1):
            doc_id = doc.page_content
            docs_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + (weights[0] * (1.0 / (idx + rrf_k)))
            
        for idx, doc in enumerate(res_chroma, start=1):
            doc_id = doc.page_content
            docs_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + (weights[1] * (1.0 / (idx + rrf_k)))
            
        # Ordenar por score RRF
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        resultados_raw = [docs_map[doc_id] for doc_id, score in sorted_scores]
        
        filas = []
        for rank, (doc_id, score) in enumerate(sorted_scores, start=1):
            doc = docs_map[doc_id]
            filas.append({
                "rank": rank,
                "score_distancia": round(score, 6),
                "id": doc.metadata.get("id"),
                "tipo_documento": doc.metadata.get("tipo_documento"),
                "documento": doc.metadata.get("documento"),
                "seccion": doc.metadata.get("seccion"),
                "catalogo": doc.metadata.get("catalogo"),
                "texto": doc.page_content[:500]
            })

        return pd.DataFrame(filas), resultados_raw

    def buscar_bm25(self, query: str, documentos_bm25: list, k: int = 5) -> tuple[pd.DataFrame, list]:
        """
        Ejecuta Búsqueda Léxica pura usando BM25.
        """
        from langchain_community.retrievers import BM25Retriever
        
        print(f"Ejecutando Búsqueda Léxica (BM25) para: '{query}'...")
        bm25_retriever = BM25Retriever.from_documents(documentos_bm25)
        bm25_retriever.k = k
        
        resultados_raw = bm25_retriever.invoke(query)
        
        filas = []
        for rank, doc in enumerate(resultados_raw, start=1):
            filas.append({
                "rank": rank,
                "score_distancia": "N/A (BM25)", 
                "id": doc.metadata.get("id"),
                "tipo_documento": doc.metadata.get("tipo_documento"),
                "documento": doc.metadata.get("documento"),
                "texto": doc.page_content[:500]
            })

        return pd.DataFrame(filas), resultados_raw

    def buscar_bow(self, query: str, documentos_raw: list, k: int = 5) -> tuple[pd.DataFrame, list]:
        """Búsqueda Léxica pura usando Bag of Words (CountVectorizer)."""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import nltk
        from nltk.corpus import stopwords
        
        try:
            stopwords_es = stopwords.words("spanish")
        except LookupError:
            nltk.download('stopwords')
            stopwords_es = stopwords.words("spanish")
        
        print(f"Ejecutando Búsqueda Léxica (BoW) para: '{query}'...")
        textos = [doc.page_content for doc in documentos_raw]
        vectorizer = CountVectorizer(lowercase=True, stop_words=stopwords_es)
        X = vectorizer.fit_transform(textos)
        
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, X).ravel()
        top_idx = sims.argsort()[::-1][:k]
        
        docs_ordenados = [documentos_raw[i] for i in top_idx]
        
        filas = []
        for rank, (doc, score) in enumerate(zip(docs_ordenados, sims[top_idx]), start=1):
            filas.append({
                "rank": rank,
                "score_distancia": float(score),
                "id": doc.metadata.get("id"),
                "tipo_documento": doc.metadata.get("tipo_documento"),
                "documento": doc.metadata.get("documento"),
                "texto": doc.page_content[:500]
            })
        return pd.DataFrame(filas), docs_ordenados

    def buscar_tfidf(self, query: str, documentos_raw: list, k: int = 5) -> tuple[pd.DataFrame, list]:
        """Búsqueda Léxica usando TF-IDF con normalización L2."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import nltk
        from nltk.corpus import stopwords
        
        try:
            stopwords_es = stopwords.words("spanish")
        except LookupError:
            nltk.download('stopwords')
            stopwords_es = stopwords.words("spanish")
        
        print(f"Ejecutando Búsqueda Léxica (TF-IDF) para: '{query}'...")
        textos = [doc.page_content for doc in documentos_raw]
        vectorizer = TfidfVectorizer(lowercase=True, stop_words=stopwords_es, norm='l2')
        X = vectorizer.fit_transform(textos)
        
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, X).ravel()
        top_idx = sims.argsort()[::-1][:k]
        
        docs_ordenados = [documentos_raw[i] for i in top_idx]
        
        filas = []
        for rank, (doc, score) in enumerate(zip(docs_ordenados, sims[top_idx]), start=1):
            filas.append({
                "rank": rank,
                "score_distancia": float(score),
                "id": doc.metadata.get("id"),
                "tipo_documento": doc.metadata.get("tipo_documento"),
                "documento": doc.metadata.get("documento"),
                "texto": doc.page_content[:500]
            })
        return pd.DataFrame(filas), docs_ordenados

    def buscar_hibrido_reranking(self, query: str, documentos_bm25: list, k: int = 5, top_n_rerank: int = 15) -> tuple[pd.DataFrame, list]:
        """
        Ejecuta Búsqueda Semántica recuperando un Top N amplio, y luego reordena los 
        resultados usando un Cross-Encoder (sentence-transformers).
        """
        from sentence_transformers.cross_encoder import CrossEncoder
        import numpy as np

        print(f"Ejecutando Embeddings + Cross-Encoder Reranking para: '{query}'...")
        
        # 1. Recuperar amplio espectro (evitando envenenamiento léxico)
        raw_docs = self.vectorstore.similarity_search(query, k=top_n_rerank)
        
        # 2. Cargar Cross-Encoder
        # Se usa un modelo ligero multilingüe o estándar
        modelo_ce = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
        encoder = CrossEncoder(modelo_ce)
        
        # 3. Preparar pares (Query, Documento)
        pares = [[query, doc.page_content] for doc in raw_docs]
        
        # 4. Calcular scores de relevancia
        scores = encoder.predict(pares)
        
        # 5. Reordenar documentos según el score de mayor a menor
        indices_ordenados = np.argsort(scores)[::-1][:k]
        docs_ordenados = [raw_docs[i] for i in indices_ordenados]
        
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

    def buscar_multi_query(self, query: str, k: int = 5, documentos_bm25: list = None, weights: list = [0.5, 0.5]) -> tuple[pd.DataFrame, list]:
        """
        Ejecuta Búsqueda Semántica utilizando MultiQueryExpansion.
        Si se proporcionan `documentos_bm25`, utiliza Búsqueda Híbrida (EnsembleRetriever)
        como motor de búsqueda para cada query generada. Si no, usa solo ChromaDB.
        """
        from src.nlp_core.config_llm import get_langchain_chat
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        print(f"Ejecutando Multi-Query Expansion para: '{query}'...")

        # 1. Configurar LLM para la generación de queries
        llm = get_langchain_chat(task="expansion", temperature=0)

        # 2. Configurar el retriever base (Chroma o Híbrido)
        chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        if documentos_bm25:
            print(" -> Integrando Búsqueda Híbrida (BM25 + Chroma) manual para cada query generada...")
            from langchain_community.retrievers import BM25Retriever
            
            bm25_retriever = BM25Retriever.from_documents(documentos_bm25)
            bm25_retriever.k = k
            
            base_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever],
                weights=weights
            )
        else:
            base_retriever = chroma_retriever

        # 3. Ensamblar MultiQueryRetriever envolviendo al retriever base
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, llm=llm
        )

        # 4. Obtener resultados consolidados
        resultados_raw = multi_query_retriever.invoke(query)[:k]

        # 5. Formatear como DataFrame
        filas = []
        for rank, doc in enumerate(resultados_raw, start=1):
            filas.append({
                "rank": rank,
                "score_distancia": "N/A (Multi-Query)", 
                "id": doc.metadata.get("id"),
                "tipo_documento": doc.metadata.get("tipo_documento"),
                "documento": doc.metadata.get("documento"),
                "seccion": doc.metadata.get("seccion"),
                "catalogo": doc.metadata.get("catalogo"),
                "texto": doc.page_content[:500]
            })

        return pd.DataFrame(filas), resultados_raw

    # ==========================================
    # MÉTODOS ESTÁTICOS PARA TF-IDF E HÍBRIDOS
    # ==========================================

    @staticmethod
    def top_tfidf_terms(vectorizer, X, top_n=25) -> pd.DataFrame:
        """Extrae los términos con mayor peso TF-IDF."""
        feature_names = np.array(vectorizer.get_feature_names_out())
        scores = np.asarray(X.mean(axis=0)).ravel()
        top_idx = scores.argsort()[::-1][:top_n]

        return pd.DataFrame({
            "termino": feature_names[top_idx],
            "score_promedio_tfidf": scores[top_idx]
        })

    @staticmethod
    def buscar_tfidf_estatico(query: str, vectorizer, X, df_base: pd.DataFrame, k: int = 5) -> pd.DataFrame:
        """
        Busca usando similitud del coseno sobre la matriz TF-IDF.
        """
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, X).ravel()
        top_idx = sims.argsort()[::-1][:k]

        resultados = df_base.iloc[top_idx].copy()
        resultados["score_similitud"] = sims[top_idx]

        return resultados[["id", "tipo_documento", "documento", "seccion", "catalogo", "score_similitud", "texto"]]

    @staticmethod
    def resumen_resultados_busqueda(nombre_consulta: str, resultados_tfidf: pd.DataFrame, resultados_chroma: pd.DataFrame) -> pd.DataFrame:
        """
        Construye una comparación simple entre los tipos de documentos recuperados por TF-IDF y ChromaDB.
        """
        if not resultados_tfidf.empty and 'tipo_documento' in resultados_tfidf.columns:
            tfidf_counts = resultados_tfidf["tipo_documento"].value_counts().reset_index()
            tfidf_counts.columns = ['tipo_documento', 'conteo_tfidf']
        else:
            tfidf_counts = pd.DataFrame(columns=['tipo_documento', 'conteo_tfidf'])

        if not resultados_chroma.empty and 'tipo_documento' in resultados_chroma.columns:
            chroma_counts = resultados_chroma["tipo_documento"].value_counts().reset_index()
            chroma_counts.columns = ['tipo_documento', 'conteo_chroma']
        else:
            chroma_counts = pd.DataFrame(columns=['tipo_documento', 'conteo_chroma'])

        resumen = pd.merge(
            tfidf_counts,
            chroma_counts,
            on="tipo_documento",
            how="outer"
        ).fillna(0)

        resumen["consulta"] = nombre_consulta
        return resumen
