import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys
import os

# Configurar PYTHONPATH para que encuentre 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.lab.diversidad_eval import calcular_diversidad

st.set_page_config(page_title="Dashboard Analítico MLOps", page_icon="📊", layout="wide")
st.title("📊 Laboratorio Analítico MLOps (Banxico DISF)")
st.markdown("Dashboard interactivo para monitorear rendimiento, telemetría y seguridad de arquitecturas RAG (Avance 5).")

# Root path
project_root = Path(__file__).resolve().parents[1]
eval_dir = project_root / "data" / "03_output" / "evaluaciones" / "oficiales"

if not eval_dir.exists():
    eval_dir = project_root / "data" / "03_output" / "evaluaciones"

# Descubrir corridas
available_runs = [d.name for d in eval_dir.iterdir() if d.is_dir()] if eval_dir.exists() else []

st.sidebar.header("⚙️ Comparador de Modelos")
if not available_runs:
    st.sidebar.warning("⚠️ No se encontraron resultados. Ejecuta el evaluador primero.")

selected_runs = st.sidebar.multiselect(
    "Selecciona Corridas (Nube vs Local):",
    options=available_runs,
    default=available_runs[:1] if available_runs else []
)

# --- DATA LOADERS (NUEVA LÓGICA AVANCE 5) ---
@st.cache_data
def load_arena_data(runs):
    resultados = []
    for run in runs:
        run_path = eval_dir / run
        csv_files = list(run_path.glob("ARENA_RESULTADOS*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            for _, row in df.iterrows():
                modelo_id = str(row.get("Candidato_ID", row.get("estrategia", row.iloc[0]))) + f" [{run}]"
                costo_total = float(row.get("Costo_Total_USD", row.get("Costo", 0.0)))
                
                # Soportar formatos viejos y nuevos
                ndcg_val = float(row.get("NDCG_Promedio", row.get("NDCG@10", row.get("NDCG_Mean", 0.0))))
                lat_p95 = float(row.get("Latencia_P95_seg", 0.0))
                
                resultados.append({
                    "Modelo": modelo_id,
                    "Corrida": run,
                    "Costo_Operativo": costo_total,
                    "NDCG_10": ndcg_val,
                    "Latencia_P95": lat_p95
                })
    return pd.DataFrame(resultados)

@st.cache_data
def load_taxonomy_data(runs):
    errores = []
    for run in runs:
        run_path = eval_dir / run
        # Buscar tanto el nuevo "resultados_llm_judge" como el viejo "analisis_errores"
        archivos_juez = list(run_path.glob("resultados_llm_judge*.csv")) + list(run_path.glob("analisis_errores*.csv"))
        for archivo in archivos_juez:
            df = pd.read_csv(archivo)
            col_error = None
            if 'Clasificacion_Error' in df.columns:
                col_error = 'Clasificacion_Error'
            elif 'Tipo_Error' in df.columns:
                col_error = 'Tipo_Error'
            elif 'Categoría Error (A/B/C)' in df.columns:
                col_error = 'Categoría Error (A/B/C)'
                
            if col_error:
                conteo = df[col_error].value_counts().reset_index()
                conteo.columns = ['Tipo_Error', 'Cantidad']
                conteo['Archivo'] = archivo.name
                conteo['Corrida'] = run
                errores.append(conteo)
    if errores:
        return pd.concat(errores, ignore_index=True)
    return pd.DataFrame()

df_arena = load_arena_data(selected_runs)
df_errores = load_taxonomy_data(selected_runs)

# --- PESTAÑAS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Pareto y Telemetría (ENS-B/C)", 
    "🧩 Taxonomía Errores (MA6)", 
    "🔀 Diversidad (ENS-D)", 
    "🛡️ Red-Teaming (DEP-D)"
])

# --- PESTAÑA 1: Frontera de Pareto (NUEVA VERSIÓN PLOTLY INTERACTIVA) ---
with tab1:
    st.header("📈 Optimización Multiobjetivo (Frontera de Pareto)")
    if df_arena.empty:
        st.info("Selecciona al menos una corrida en la barra lateral.")
    else:
        df_sorted = df_arena.sort_values(by=["Costo_Operativo", "NDCG_10"], ascending=[True, False])
        
        frontera_x, frontera_y = [], []
        max_ndcg = -1.0
        for _, row in df_sorted.iterrows():
            if row["NDCG_10"] > max_ndcg:
                frontera_x.append(row["Costo_Operativo"])
                frontera_y.append(row["NDCG_10"])
                max_ndcg = row["NDCG_10"]
                
        fig = px.scatter(
            df_arena, x="Costo_Operativo", y="NDCG_10", color="Corrida",
            hover_name="Modelo", size_max=15,
            title="Frontera de Pareto (Interactivo)"
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        
        fig.add_trace(go.Scatter(
            x=frontera_x, y=frontera_y, mode='lines', name='Frontera de Pareto',
            line=dict(color='red', width=2, dash='dash')
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Tabla Resumen (Telemetría de Ensambles)")
        st.dataframe(
            df_arena.sort_values("NDCG_10", ascending=False),
            use_container_width=True,
            column_config={
                "Costo_Operativo": st.column_config.NumberColumn("Costo (USD)", format="$%.4f"),
                "Latencia_P95": st.column_config.NumberColumn("Latencia P95", format="%.2f s"),
                "NDCG_10": st.column_config.NumberColumn("NDCG@10", format="%.3f")
            }
        )

# --- PESTAÑA 2: Desagregación de Errores (NUEVA VERSIÓN MULTI-CORRIDA) ---
with tab2:
    st.header("🧩 Taxonomía de Fallos (A/B/C)")
    if df_errores.empty:
        st.warning("No se encontraron resultados de validación de Errores.")
    else:
        archivos_disponibles = df_errores["Archivo"].unique()
        archivo_sel = st.selectbox("Selecciona un candidato para ver su taxonomía:", archivos_disponibles)
        
        df_filtrado = df_errores[df_errores["Archivo"] == archivo_sel]
        fig_bar = px.bar(
            df_filtrado, x="Tipo_Error", y="Cantidad", color="Tipo_Error",
            title=f"Distribución de Errores - {archivo_sel}", text="Cantidad",
            color_discrete_map={'A':'#EF4444', 'B':'#F59E0B', 'C':'#3B82F6'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --- PESTAÑA 3: Diversidad (LÓGICA ORIGINAL PRESERVADA) ---
with tab3:
    st.header("🔀 Diversidad Cuantificada (Disagreement Rate)")
    if not selected_runs:
        st.info("Selecciona una corrida.")
    else:
        run_path = eval_dir / selected_runs[0]
        archivos_juez = list(run_path.glob("resultados_llm_judge_*.csv")) + list(run_path.glob("analisis_errores_desagregados_*.csv"))
        
        if len(archivos_juez) >= 2:
            modelos = [f.stem for f in archivos_juez]
            mod_a = st.selectbox("Modelo A", modelos, index=0)
            mod_b = st.selectbox("Modelo B", modelos, index=1)
            
            if st.button("Calcular Diversidad"):
                ruta_a = [f for f in archivos_juez if f.stem == mod_a][0]
                ruta_b = [f for f in archivos_juez if f.stem == mod_b][0]
                
                try:
                    res_div = calcular_diversidad(str(ruta_a), str(ruta_b), mod_a, mod_b)
                    if res_div:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Disagreement Rate", f"{res_div['metricas_ens_d']['disagreement_rate']*100:.1f}%")
                        col2.metric("Correlación Errores", f"{res_div['metricas_ens_d']['error_correlation']:.3f}")
                        col3.metric("Oracle Gap", f"+{res_div['metricas_ens_d']['oracle_gap']*100:.1f}%")
                        
                        st.subheader("Matriz de Acuerdo")
                        matriz = res_div['matriz_acuerdo']
                        df_mat = pd.DataFrame({
                            f"{mod_b} Acertó": [matriz['ambos_aciertan'], matriz[f'solo_{mod_b}_acierta']],
                            f"{mod_b} Falló": [matriz[f'solo_{mod_a}_acierta'], matriz['ambos_fallan']]
                        }, index=[f"{mod_a} Acertó", f"{mod_a} Falló"])
                        st.table(df_mat)
                except Exception as e:
                    st.error(f"Error calculando diversidad: {e}")
        else:
            st.warning("Se necesitan al menos 2 modelos para medir diversidad.")

# --- PESTAÑA 4: Red-Teaming (LÓGICA ORIGINAL PRESERVADA) ---
with tab4:
    st.header("🛡️ Red-Teaming y Seguridad (DEP-D)")
    ruta_reporte_rt = project_root / "data" / "03_output" / "evaluaciones" / "red_teaming_reporte.json"
    
    if ruta_reporte_rt.exists():
        with open(ruta_reporte_rt, "r", encoding="utf-8") as f:
            reporte = json.load(f)
        st.metric("Tasa de Defensa", f"{reporte['tasa_exito_defensiva']}%")
        df_rt = pd.DataFrame(reporte['detalles'])
        
        def highlight_veredicto(val):
            color = 'lightgreen' if val == 'BLOCKED' else 'lightcoral'
            return f'background-color: {color}'
            
        st.dataframe(df_rt.style.map(highlight_veredicto, subset=['veredicto']))
    else:
        st.info("No se ha ejecutado el módulo de seguridad (`src/lab/seguridad_eval.py`).")
