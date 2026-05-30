import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Configurar PYTHONPATH para que encuentre 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.lab.graficos import plot_frontera_pareto
from src.lab.diversidad_eval import calcular_diversidad

# Configuración de página
st.set_page_config(page_title="Dashboard de Evaluaciones", page_icon="📊", layout="wide")
st.title("📊 Laboratorio Analítico de Modelos (Banxico DISF)")
st.markdown("Este dashboard interactivo permite monitorear y comparar el rendimiento, seguridad y costos de los modelos candidatos, cumpliendo con las rúbricas E5 y E6.")

# Encontrar la carpeta de evaluación más reciente
def get_latest_run(base_path: str):
    p = Path(base_path)
    if not p.exists(): return None
    carpetas = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not carpetas: return None
    carpetas.sort(key=lambda x: x.name, reverse=True)
    return carpetas[0]

# Intentamos cargar de 'oficiales' primero, sino 'pruebas_rapidas'
ruta_base_oficial = "data/03_output/evaluaciones/oficiales"
ruta_base_rapidas = "data/03_output/evaluaciones/pruebas_rapidas"

latest_run = get_latest_run(ruta_base_oficial)
if not latest_run:
    latest_run = get_latest_run(ruta_base_rapidas)

if not latest_run:
    st.warning("⚠️ No se encontraron resultados de evaluaciones. Corre `python src/lab/evaluador_integral.py` primero.")
    st.stop()

st.success(f"Cargando datos de la corrida más reciente: **{latest_run.name}**")

# Pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Frontera de Pareto (MA7/ENS-C)", 
    "🧩 Desagregación de Errores (MA6)", 
    "🔀 Diversidad y Consistencia (ENS-D/E)", 
    "🛡️ Red-Teaming (DEP-D)"
])

# --- PESTAÑA 1: Frontera de Pareto ---
with tab1:
    st.header("📈 Frontera de Pareto (Costo/Latencia vs Precisión)")
    st.markdown("Balance entre costo de inferencia (o latencia) y la métrica principal **NDCG@10**.")
    
    if st.button("Generar Gráfica de Pareto"):
        with st.spinner("Calculando y renderizando..."):
            archivos_arena = list(latest_run.glob("ARENA_RESULTADOS*.csv"))
            if not archivos_arena:
                st.error("No se encontró el CSV ARENA_RESULTADOS para graficar.")
            else:
                df_arena = pd.read_csv(archivos_arena[0])
                resultados_grafico = []
                for _, row in df_arena.iterrows():
                    modelo = row['estrategia']
                    ndcg = row.get('NDCG@10', 0.0)
                    
                    # Estimar costo
                    tokens = row.get('Tokens_Contexto_Promedio', 2000)
                    es_local = row.get('Es_QA_Local', True)
                    costo_1000 = 0.0
                    if not es_local:
                        costo_1000 = (tokens / 1000000.0) * 0.15 * 1000.0 # Aproximación simple usando precios de gpt-4o-mini
                        
                    resultados_grafico.append({
                        "modelo": modelo,
                        "costo_por_1000": costo_1000,
                        "ndcg": ndcg
                    })
                
                output_path = str(Path("data/03_output/pareto_frontier.png").absolute())
                plot_frontera_pareto(resultados_grafico, output_path)
                
                if Path(output_path).exists():
                    st.image(output_path, use_container_width=True)
                else:
                    st.error("No se pudo generar la gráfica de Pareto.")
                
    # Mostrar tabla resumen si existe
    archivos_arena = list(latest_run.glob("ARENA_RESULTADOS*.csv"))
    if archivos_arena:
        st.subheader("Tabla Resumen (Ensembles y Baselines)")
        df_arena = pd.read_csv(archivos_arena[0])
        st.dataframe(df_arena)

# --- PESTAÑA 2: Desagregación de Errores ---
with tab2:
    st.header("🧩 Taxonomía de Fallos Automática")
    st.markdown("Clasifica los errores en: **A** (Fallo Retrieval), **B** (Fallo Generación/Alucinación), **C** (Fallo de Formato Estricto).")
    
    archivos_err = list(latest_run.glob("analisis_errores*.csv"))
    if archivos_err:
        for archivo in archivos_err:
            df_err = pd.read_csv(archivo)
            if 'Categoría Error (A/B/C)' in df_err.columns:
                conteo = df_err['Categoría Error (A/B/C)'].value_counts().reset_index()
                conteo.columns = ['Categoría', 'Cantidad']
                
                fig = px.pie(conteo, values='Cantidad', names='Categoría', 
                             title=f"Distribución de Errores - {archivo.stem.replace('analisis_errores_desagregados_', '')}",
                             color='Categoría',
                             color_discrete_map={'A':'#EF4444', 'B':'#F59E0B', 'C':'#3B82F6'})
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_err)
    else:
        st.info("No se detectaron errores en esta corrida (¡Perfecto!) o no se generó el archivo de análisis.")

# --- PESTAÑA 3: Diversidad y Consistencia ---
with tab3:
    st.header("🔀 Diversidad Cuantificada (ENS-D)")
    st.markdown("Compara dos modelos para ver en qué se equivocan y si sus errores están correlacionados (Matriz de Acuerdo).")
    
    archivos_juez = list(latest_run.glob("resultados_llm_judge_*.csv"))
    if len(archivos_juez) >= 2:
        modelos = [f.stem for f in archivos_juez]
        mod_a = st.selectbox("Selecciona Modelo A", modelos, index=0)
        mod_b = st.selectbox("Selecciona Modelo B", modelos, index=1)
        
        if st.button("Calcular Matriz de Diversidad"):
            ruta_a = [f for f in archivos_juez if f.stem == mod_a][0]
            ruta_b = [f for f in archivos_juez if f.stem == mod_b][0]
            
            res_div = calcular_diversidad(str(ruta_a), str(ruta_b), mod_a, mod_b)
            if res_div:
                col1, col2, col3 = st.columns(3)
                col1.metric("Disagreement Rate", f"{res_div['metricas_ens_d']['disagreement_rate']*100:.1f}%")
                col2.metric("Correlación de Errores", f"{res_div['metricas_ens_d']['error_correlation']:.3f}")
                col3.metric("Oracle Gap (Mejora Máxima)", f"+{res_div['metricas_ens_d']['oracle_gap']*100:.1f}%")
                
                # Matriz de Confusión / Acuerdo
                st.subheader("Matriz de Acuerdo")
                matriz = res_div['matriz_acuerdo']
                df_mat = pd.DataFrame({
                    f"{mod_b} Acertó": [matriz['ambos_aciertan'], matriz[f'solo_{mod_b}_acierta']],
                    f"{mod_b} Falló": [matriz[f'solo_{mod_a}_acierta'], matriz['ambos_fallan']]
                }, index=[f"{mod_a} Acertó", f"{mod_a} Falló"])
                st.table(df_mat)
    else:
        st.warning("Se necesitan al menos 2 modelos evaluados para medir diversidad.")

# --- PESTAÑA 4: Red-Teaming y Seguridad ---
with tab4:
    st.header("🛡️ Pruebas de Seguridad y Red-Teaming (DEP-D)")
    st.markdown("Mide la robustez del modelo frente a inyección de prompts, extracción de PII y jailbreaks.")
    
    ruta_reporte_rt = Path("data/03_output/evaluaciones/red_teaming_reporte.json")
    if ruta_reporte_rt.exists():
        with open(ruta_reporte_rt, "r", encoding="utf-8") as f:
            reporte = json.load(f)
            
        st.metric("Tasa de Defensa (Guardrails Exitosos)", f"{reporte['tasa_exito_defensiva']}%")
        
        df_rt = pd.DataFrame(reporte['detalles'])
        
        # Colorear fila dependiendo del veredicto
        def highlight_veredicto(val):
            color = 'lightgreen' if val == 'BLOCKED' else 'lightcoral'
            return f'background-color: {color}'
            
        st.dataframe(df_rt.style.map(highlight_veredicto, subset=['veredicto']))
    else:
        st.info("No se ha corrido el laboratorio de seguridad. Ejecuta `python src/lab/seguridad_eval.py` para poblar este reporte.")
