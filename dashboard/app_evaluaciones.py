import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
import sys
import os
import time

# Configurar PYTHONPATH para que encuentre 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.nlp_core.generacion import responder_rag_cascade_qa
from src.nlp_core.evals.evaluador import evaluar_faithfulness_claims, evaluar_answer_relevance, evaluar_context_relevancy
from src.lab.consistencia_eval import correr_evaluacion_consistencia

st.set_page_config(page_title="Centro de Comando MLOps", page_icon="🚀", layout="wide")
st.title("🚀 Centro de Comando MLOps (Banxico DISF)")
st.markdown("Monitoreo en Vivo y Evaluación bajo demanda de la versión en Producción (Cascade Router).")

# Root path
project_root = Path(__file__).resolve().parents[1]

# --- DATA LOADER TELEMETRÍA ---
@st.cache_data(ttl=5) # Refrescar cada 5 segundos
def load_telemetry_in_vivo():
    telemetria_path = project_root / "data" / "03_output" / "telemetria_llm.jsonl"
    if telemetria_path.exists():
        try:
            df = pd.read_json(telemetria_path, lines=True)
            # Calcular costos dinámicamente si no están
            def calcular_costo(row):
                mod = str(row.get('modelo', '')).lower()
                in_tok = row.get('prompt_tokens', 0)
                out_tok = row.get('completion_tokens', 0)
                if 'mini' in mod:
                    return (in_tok / 1e6) * 0.15 + (out_tok / 1e6) * 0.60
                elif 'gpt-4o' in mod:
                    return (in_tok / 1e6) * 5.0 + (out_tok / 1e6) * 15.0
                return 0.0 # Llama local u otro
                
            df['costo_estimado_usd'] = df.apply(calcular_costo, axis=1)
            # Asegurar datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            st.error(f"Error leyendo telemetría en vivo: {e}")
    return pd.DataFrame()

df_telemetria = load_telemetry_in_vivo()



# --- NAVEGACIÓN LATERAL ---
st.sidebar.title("Navegación")
menu = st.sidebar.radio(
    "Selecciona un módulo:",
    ["⚙️ Estado del Despliegue", "🧪 Pruebas RAGAS (Bajo Demanda)", "📡 Monitoreo Operativo en Vivo"]
)

# --- MÓDULO 1: Estado del Despliegue ---
if menu == "⚙️ Estado del Despliegue":
    
    import inspect
    sig = inspect.signature(responder_rag_cascade_qa)
    def_motor = sig.parameters['base_retriever'].default
    def_exp = sig.parameters['query_expansion'].default
    def_post = sig.parameters['post_processing'].default
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Motor Base", str(def_motor).capitalize())
    col2.metric("Expansión Query", str(def_exp).capitalize())
    col3.metric("Post-procesamiento", str(def_post).capitalize())
    
    st.info("El sistema primero intenta responder con **Llama 3.1 (Local)**. Si evalúa que la respuesta tiene un *Faithfulness Score* menor a 0.8, redirige la pregunta automáticamente a **GPT-4o (Nube)**.")

# --- MÓDULO 2: Pruebas Bajo Demanda ---
elif menu == "🧪 Pruebas RAGAS (Bajo Demanda)":
    st.header("🧪 Pruebas RAGAS sobre la API en Producción")
    st.markdown("Escribe una consulta para probar en vivo la respuesta del modelo, y calcularemos sus métricas RAGAS al vuelo.")
    
    query = st.text_input("Consulta Normativa:", "¿Cuándo se debe enviar el reporte de liquidez a la CNBV?")
    
    if st.button("Ejecutar Consulta y Evaluar RAGAS"):
        with st.spinner("Ejecutando Cascade Pipeline y evaluando..."):
            t_inicio = time.time()
            try:
                # 1. Llamar a la función principal del backend
                respuesta, meta, chunks = responder_rag_cascade_qa(query)
                latencia_total = time.time() - t_inicio
                
                contexto_str = "\n".join([c["content"] for c in chunks])
                
                # 2. RAGAS On-the-fly
                score_faithfulness = meta.get("faithfulness_local_score") # Ya lo calcula el Cascade
                if score_faithfulness is None:
                    score_faithfulness = evaluar_faithfulness_claims(respuesta, contexto_str)
                    
                score_relevancia = evaluar_answer_relevance(query, respuesta)
                score_contexto = evaluar_context_relevancy(query, contexto_str)
                
                # Guardar en session state para que no desaparezca al interactuar
                st.session_state["ragas_eval_result"] = {
                    "latencia_total": latencia_total,
                    "respuesta": respuesta,
                    "score_faithfulness": score_faithfulness,
                    "score_relevancia": score_relevancia,
                    "score_contexto": score_contexto,
                    "estrategia": meta.get("estrategia_cascade", "Directo"),
                    "contexto_str": contexto_str
                }
                    
            except Exception as e:
                st.error(f"Error durante la prueba: {e}")

    # Mostrar resultados si existen en el estado
    if "ragas_eval_result" in st.session_state:
        res = st.session_state["ragas_eval_result"]
        st.success(f"¡Ejecución completa en {res['latencia_total']:.2f}s!")
        
        # Mostrar Resultados
        st.subheader("🤖 Respuesta Generada")
        st.markdown(res['respuesta'])
        
        st.subheader("⚖️ Resultados RAGAS (Reference-Free)")
        c1, c2, c3, c4 = st.columns(4)
        
        c1.metric("Faithfulness Score", f"{res['score_faithfulness']:.2f}", help="¿Es la respuesta fiel al contexto o alucina?")
        c2.metric("Answer Relevance", f"{res['score_relevancia']:.2f}", help="¿Responde directamente a la pregunta original?")
        c3.metric("Context Relevancy", f"{res['score_contexto']:.2f}", help="¿Qué tan útil es el contexto recuperado?")
        
        estrategia = res['estrategia']
        color = "green" if "Local" in estrategia else "orange"
        if "Bloqueado" in estrategia:
            color = "red"
        c4.markdown(f"**Estrategia Final:** <span style='color:{color};'>{estrategia}</span>", unsafe_allow_html=True)
        
        with st.expander("Ver Contexto Recuperado"):
            st.markdown(res['contexto_str'])

    st.divider()
    st.markdown("### 🎲 Prueba de Miscalibración y Alucinación Encubierta")
    st.markdown("Esta prueba de estrés ('Self-Consistency') tomará la consulta actual y la lanzará al modelo de generación **3 veces seguidas con alta temperatura (0.7)**. Luego, un juez LLM evaluará si el modelo fue capaz de mantener una narrativa coherente o si comenzó a contradecirse (lo que indicaría alucinación encubierta).")
    
    if st.button("Ejecutar Prueba de Consistencia (ECE)"):
        with st.spinner("Ejecutando 3 corridas con temperatura 0.7 y calculando varianza... Esto puede tomar varios segundos."):
            try:
                res_ece = correr_evaluacion_consistencia(query, n_runs=3, temperatura=0.7)
                st.success("¡Prueba de consistencia completada!")
                
                col_e1, col_e2, col_e3, col_e4 = st.columns(4)
                col_e1.metric("Consistency Score", f"{res_ece['consistency_score']*100:.1f}%")
                col_e2.metric("Varianza", f"{res_ece['varianza']:.4f}")
                col_e3.metric("ECE (Miscalibración)", f"{res_ece['ece_aprox']:.4f}")
                
                color_diag = "green" if "Consistente" in res_ece['interpretacion'] else "red"
                col_e4.markdown(f"**Diagnóstico:** <span style='color:{color_diag};'>{res_ece['interpretacion']}</span>", unsafe_allow_html=True)
                
                # Mostrar las respuestas generadas
                with st.expander("Ver las 3 respuestas generadas (Self-Consistency)"):
                    for idx, resp in enumerate(res_ece.get("respuestas", [])):
                        st.markdown(f"**Corrida {idx+1}:**")
                        st.info(resp)
            except Exception as e:
                st.error(f"Error durante la prueba de consistencia: {e}")

# --- MÓDULO 3: Telemetría en Vivo ---
elif menu == "📡 Monitoreo Operativo en Vivo":
    st.header("📡 Monitoreo Operativo y FinOps")
    
    # Botón de refresco manual
    if st.button("🔄 Refrescar Datos en Vivo"):
        st.cache_data.clear()
        st.rerun()
        
    if df_telemetria.empty:
        st.info("No hay datos de telemetría en vivo. Interactúa con la aplicación web para generar logs.")
    else:
        # 1. KPIs
        costo_total = df_telemetria['costo_estimado_usd'].sum()
        total_consultas = len(df_telemetria)
        avg_latencia = df_telemetria['latencia_total_seg'].mean() if 'latencia_total_seg' in df_telemetria.columns else 0
        
        # Promedio Faithfulness
        avg_faith = df_telemetria['faithfulness_local_score'].mean() if 'faithfulness_local_score' in df_telemetria.columns else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Costo Total Acumulado", f"${costo_total:.4f} USD")
        c2.metric("Consultas Procesadas", f"{total_consultas}")
        c3.metric("Latencia Promedio", f"{avg_latencia:.2f} s")
        c4.metric("Faithfulness Promedio", f"{avg_faith:.2f}")
        
        st.divider()
        
        col_graf_1, col_graf_2 = st.columns(2)
        
        # 2. Distribución de Cascade
        with col_graf_1:
            st.subheader("Rutas de Cascade Router")
            if 'estrategia_cascade' in df_telemetria.columns:
                # Contar ocurrencias
                conteo_estrategias = df_telemetria['estrategia_cascade'].value_counts().reset_index()
                conteo_estrategias.columns = ['Estrategia', 'Cantidad']
                
                fig_pie = px.pie(conteo_estrategias, names='Estrategia', values='Cantidad', 
                                 title="Porcentaje de Consultas resueltas Local vs Nube",
                                 color='Estrategia',
                                 color_discrete_map={'Resuelto Local':'#2ca02c', 'Escalado a Nube':'#ff7f0e'})
                st.plotly_chart(fig_pie, use_container_width=True)
                
        # 3. Evolución del Costo
        with col_graf_2:
            st.subheader("Evolución del Costo FinOps")
            if 'timestamp' in df_telemetria.columns and 'costo_estimado_usd' in df_telemetria.columns:
                df_temp = df_telemetria.sort_values('timestamp').copy() # Evitar warning de Pandas
                df_temp['Costo_Acumulado'] = df_temp['costo_estimado_usd'].cumsum()
                fig_line = px.line(df_temp, x="timestamp", y="Costo_Acumulado", markers=True,
                                   title="Costo Operativo Acumulado (USD)")
                st.plotly_chart(fig_line, use_container_width=True)
                
        st.subheader("Registro Bruto de Operaciones")
        # Mostrar las columnas más relevantes
        columnas_mostrar = ['timestamp', 'estrategia', 'estrategia_cascade', 'latencia_total_seg', 'faithfulness_local_score', 'total_tokens', 'costo_estimado_usd']
        columnas_existentes = [col for col in columnas_mostrar if col in df_telemetria.columns]
        
        st.dataframe(df_telemetria.sort_values('timestamp', ascending=False)[columnas_existentes].head(50), use_container_width=True)
