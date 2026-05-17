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
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")
            
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.api_key
        )
        
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
        from langchain_classic.retrievers import EnsembleRetriever

        print(f"Ejecutando Búsqueda Híbrida para: '{query}'...")

        # 1. Configurar BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(documentos_bm25)
        bm25_retriever.k = k

        # 2. Configurar Chroma Retriever
        chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        # 3. Ensamblar usando RRF (Reciprocal Rank Fusion)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=weights
        )

        # 4. Obtener resultados combinados
        resultados_raw = ensemble_retriever.invoke(query)[:k]

        # 5. Formatear como DataFrame para consistencia
        filas = []
        for rank, doc in enumerate(resultados_raw, start=1):
            filas.append({
                "rank": rank,
                "score_distancia": "N/A (Híbrido/RRF)", 
                "id": doc.metadata.get("id"),
                "tipo_documento": doc.metadata.get("tipo_documento"),
                "documento": doc.metadata.get("documento"),
                "seccion": doc.metadata.get("seccion"),
                "catalogo": doc.metadata.get("catalogo"),
                "texto": doc.page_content[:500]
            })

        return pd.DataFrame(filas), resultados_raw

    def buscar_multi_query(self, query: str, k: int = 5, documentos_bm25: list = None, weights: list = [0.5, 0.5]) -> tuple[pd.DataFrame, list]:
        """
        Ejecuta Búsqueda Semántica utilizando MultiQueryExpansion.
        Si se proporcionan `documentos_bm25`, utiliza Búsqueda Híbrida (EnsembleRetriever)
        como motor de búsqueda para cada query generada. Si no, usa solo ChromaDB.
        """
        from langchain_openai import ChatOpenAI
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        print(f"Ejecutando Multi-Query Expansion para: '{query}'...")

        # 1. Configurar LLM para la generación de queries
        llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=self.api_key)

        # 2. Configurar el retriever base (Chroma o Híbrido)
        chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        if documentos_bm25:
            print(" -> Integrando Búsqueda Híbrida (BM25 + Chroma) para cada query generada...")
            from langchain_community.retrievers import BM25Retriever
            from langchain_classic.retrievers import EnsembleRetriever
            
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
    def buscar_tfidf(query: str, vectorizer, X, df_base: pd.DataFrame, k: int = 5) -> pd.DataFrame:
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
