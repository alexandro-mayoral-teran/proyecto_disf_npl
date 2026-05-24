# -*- coding: utf-8 -*-
"""Utilidades para presentar resultados de la Arena del Avance 3."""

from pathlib import Path
import pandas as pd
from IPython.display import Markdown, display


COLUMNAS_NUMERICAS = [
    "Recall@5",
    "Recall@10",
    "MAP@10",
    "NDCG@10",
    "Latencia_Promedio_Segundos",
    "Tokens_Contexto_Promedio",
]


def cargar_resultados(path: str | Path) -> pd.DataFrame:
    """Carga un CSV de resultados de la Arena y normaliza columnas numericas."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for col in COLUMNAS_NUMERICAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def construir_resumen_interpretativo() -> pd.DataFrame:
    """Resume el papel conceptual de cada familia de metodos evaluados."""
    return pd.DataFrame(
        [
            {
                "Familia": "Baselines lexicos",
                "Metodos": "BoW, TF-IDF, BM25",
                "Lectura": (
                    "Establecen el piso de comparacion. Son rapidos y fuertes "
                    "cuando la consulta comparte vocabulario con la norma."
                ),
                "Riesgo": (
                    "Pueden fallar ante parafrasis, formulas fragmentadas o "
                    "perdida de encabezados."
                ),
            },
            {
                "Familia": "Recuperacion semantica",
                "Metodos": "Embeddings + ChromaDB",
                "Lectura": (
                    "Captura similitud conceptual y suele mejorar cuando la "
                    "pregunta no replica literalmente el texto normativo."
                ),
                "Riesgo": (
                    "Puede recuperar fragmentos semanticamente cercanos pero "
                    "normativamente insuficientes."
                ),
            },
            {
                "Familia": "Busqueda hibrida",
                "Metodos": "BM25 + Embeddings + RRF",
                "Lectura": (
                    "Combina coincidencia exacta y similitud semantica para "
                    "reducir fallas individuales."
                ),
                "Riesgo": (
                    "Incrementa complejidad y requiere calibrar pesos o "
                    "criterios de fusion."
                ),
            },
            {
                "Familia": "Expansion y reranking",
                "Metodos": "MultiQuery, HyDE, Cross-Encoder",
                "Lectura": (
                    "Puede mejorar cobertura y ordenamiento final de candidatos."
                ),
                "Riesgo": (
                    "Aumenta latencia, costo y riesgo de introducir ruido si "
                    "la expansion no es adecuada."
                ),
            },
        ]
    )


def _preparar_tabla_resultados(df: pd.DataFrame) -> pd.DataFrame:
    columnas = [
        "estrategia",
        "modo_evaluacion",
        "Recall@5",
        "Recall@10",
        "MAP@10",
        "NDCG@10",
        "Latencia_Promedio_Segundos",
        "Tokens_Contexto_Promedio",
    ]
    columnas = [c for c in columnas if c in df.columns]
    tabla = df.sort_values(["NDCG@10", "Recall@10"], ascending=False)[columnas].copy()
    for col in ["Recall@5", "Recall@10", "Latencia_Promedio_Segundos", "Tokens_Contexto_Promedio"]:
        if col in tabla.columns:
            tabla[col] = tabla[col].round(2)
    for col in ["MAP@10", "NDCG@10"]:
        if col in tabla.columns:
            tabla[col] = tabla[col].round(4)
    return tabla


def _mostrar_tabla_resultados(df: pd.DataFrame, titulo: str) -> None:
    display(Markdown(f"### {titulo}"))
    if df.empty:
        display(Markdown("_No se encontro el archivo de resultados correspondiente._"))
        return

    # Se usa DataFrame simple en lugar de Styler para que el output se vea bien
    # en VS Code, Jupyter local y Colab.
    display(_preparar_tabla_resultados(df))


def mostrar_resultados_arena(project_root: str | Path):
    """Carga y despliega las tablas principales de resultados del Avance 3."""
    project_root = Path(project_root)
    ruta_eval = project_root / "data" / "03_output" / "evaluaciones"

    df_llm = cargar_resultados(ruta_eval / "ARENA_RESULTADOS_llm_judge.csv")
    df_exact = cargar_resultados(ruta_eval / "ARENA_RESULTADOS_exact_match.csv")
    resumen = construir_resumen_interpretativo()

    _mostrar_tabla_resultados(df_llm, "Resultados de la Arena con LLM como juez")
    _mostrar_tabla_resultados(df_exact, "Resultados de la Arena con coincidencia exacta")

    display(Markdown("### Resumen interpretativo por familia de metodos"))
    display(resumen)

    return df_llm, df_exact, resumen



def construir_analisis_componentes(df_llm: pd.DataFrame) -> pd.DataFrame:
    """Construye una lectura ablation-style de contribucion de componentes."""
    if df_llm.empty:
        return pd.DataFrame()

    metricas = df_llm.set_index("estrategia").to_dict("index")

    def fila(comparacion, componente, base, avanzado, lectura):
        base_m = metricas.get(base, {})
        adv_m = metricas.get(avanzado, {})
        base_ndcg = base_m.get("NDCG@10")
        adv_ndcg = adv_m.get("NDCG@10")
        base_recall = base_m.get("Recall@10")
        adv_recall = adv_m.get("Recall@10")
        base_lat = base_m.get("Latencia_Promedio_Segundos")
        adv_lat = adv_m.get("Latencia_Promedio_Segundos")

        return {
            "Comparacion": comparacion,
            "Componente observado": componente,
            "Metodo base": base,
            "Metodo comparado": avanzado,
            "Delta NDCG@10": None if base_ndcg is None or adv_ndcg is None else round(adv_ndcg - base_ndcg, 4),
            "Delta Recall@10": None if base_recall is None or adv_recall is None else round(adv_recall - base_recall, 2),
            "Delta latencia (s)": None if base_lat is None or adv_lat is None else round(adv_lat - base_lat, 4),
            "Lectura": lectura,
        }

    pares = [
        fila(
            "BoW -> TF-IDF",
            "Ponderacion lexica",
            "1_BoW",
            "2_TF-IDF",
            "Evalua si ponderar terminos por rareza mejora la coincidencia simple.",
        ),
        fila(
            "BM25 -> Embeddings",
            "Senal semantica",
            "3_BM25",
            "4_Embeddings",
            "Evalua si la similitud conceptual aporta sobre recuperacion lexica clasica.",
        ),
        fila(
            "Embeddings -> Hibrido RRF",
            "Fusion lexica + semantica",
            "4_Embeddings",
            "5_Hibrido_RRF",
            "Evalua si combinar senales reduce fallas individuales de cada recuperador.",
        ),
        fila(
            "Embeddings -> Embeddings + CrossEncoder",
            "Reranking",
            "4_Embeddings",
            "7_Embeddings_CrossEncoder",
            "Evalua si reordenar candidatos mejora el ranking a cambio de mayor latencia.",
        ),
        fila(
            "Embeddings + CrossEncoder -> MultiQuery + CrossEncoder",
            "Expansion de consulta",
            "7_Embeddings_CrossEncoder",
            "8_MultiQuery_Embeddings_CrossEncoder",
            "Evalua si parafrasear la consulta mejora cobertura o introduce ruido.",
        ),
        fila(
            "Embeddings + CrossEncoder -> HyDE + CrossEncoder",
            "Documento hipotetico",
            "7_Embeddings_CrossEncoder",
            "9_HyDE_Embeddings_CrossEncoder",
            "Evalua si buscar con una respuesta hipotetica mejora la recuperacion.",
        ),
    ]
    return pd.DataFrame(pares)


def mostrar_analisis_componentes_bl2(df_llm: pd.DataFrame):
    """Muestra la tabla de contribucion de componentes para BL2."""
    display(Markdown("### Analisis ablation-style de componentes (BL2)"))
    tabla = construir_analisis_componentes(df_llm)
    if tabla.empty:
        display(Markdown("_No hay resultados disponibles para construir el analisis BL2._"))
    else:
        display(tabla)
    return tabla
