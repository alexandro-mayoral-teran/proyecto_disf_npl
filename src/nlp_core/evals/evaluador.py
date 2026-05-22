import json
import math
import time
import tiktoken
import pandas as pd
from pathlib import Path

class EvaluadorRAG:
    """
    Clase dedicada a realizar pruebas cuantitativas sobre el pipeline RAG
    contra un Dataset de Evaluación (Ground Truth).
    """
    def __init__(self, ground_truth_path: str | Path):
        self.gt_path = Path(ground_truth_path)
        if not self.gt_path.exists():
            raise FileNotFoundError(f"No se encontró el Ground Truth en {self.gt_path}")
            
        with open(self.gt_path, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
            
        # Determinar el root del proyecto para guardar resultados
        self.project_root = self.gt_path.parent.parent
        self.out_dir = self.project_root / "data" / "03_output" / "evaluaciones"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self._llm_judge = None

    def _get_llm_judge(self):
        if self._llm_judge is None:
            # Usaremos gpt-4o-mini por eficiencia en la evaluación masiva
            from langchain_openai import ChatOpenAI
            self._llm_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        return self._llm_judge

    def evaluar_retrieval(self, funcion_busqueda, estrategia_nombre: str, k: int = 10, verbose: bool = True, modo_evaluacion: str = "exact_match", limite_consultas: int = None):
        """
        Evalúa una función de búsqueda específica contra el Ground Truth usando métricas clave de Information Retrieval (IR).
        
        Parámetros:
        - modo_evaluacion: "exact_match" (defecto), "human" (exporta excel para revisión manual), "llm_judge" (usa IA para evaluar hit).
        - limite_consultas: int para probar solo con las primeras N consultas (útil para pruebas de integración o debug).
        """
        if verbose:
            print(f"📊 Evaluando '{estrategia_nombre}' en modo '{modo_evaluacion}' (Top-{k})...")
            
        consultas_a_evaluar = self.ground_truth
        if limite_consultas is not None:
            consultas_a_evaluar = consultas_a_evaluar[:limite_consultas]
            if verbose:
                print(f"⚠️ Limitado a {limite_consultas} consultas para prueba rapida.")

        resultados_metricas = {
            "estrategia": estrategia_nombre,
            "hits_at_5": 0,
            "hits_at_10": 0,
            "sum_map_10": 0.0,
            "sum_ndcg_10": 0.0,
            "total_queries": len(consultas_a_evaluar)
        }
        
        registro_resultados = [] # Para el CSV normal (exact y llm)
        registro_human = []      # Para el Excel de auditoría (human)
        latencias = []
        tokens_contexto = []
        
        # Inicializar el tokenizer
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            tokenizer = None

        for item in consultas_a_evaluar:
            query = item['pregunta']
            query_id = item.get('query_id', 'N/A')
            doc_esperado = item.get('documentos_esperados', [{}])[0]
            texto_clave = doc_esperado.get('texto_clave_esperado', '')
            
            if not texto_clave:
                continue

            texto_clave_norm = " ".join(texto_clave.lower().split())
            
            # 1. Ejecutar el Retrieval
            try:
                start_time = time.time()
                resultados = funcion_busqueda(query=query, k=k)
                end_time = time.time()
                latencias.append(end_time - start_time)
            except Exception as e:
                import traceback
                print(f"[ERROR] ERROR FATAL al buscar '{query}': {e}")
                traceback.print_exc()
                continue

            if isinstance(resultados, tuple) and len(resultados) == 2:
                docs = resultados[1]
            else:
                docs = resultados
                
            texto_contexto = " ".join([doc.page_content for doc in docs])
            if tokenizer:
                tokens_contexto.append(len(tokenizer.encode(texto_contexto)))
            else:
                tokens_contexto.append(int(len(texto_contexto.split()) * 1.3))

            # 2. Evaluación del Hit (Dependiente del Modo)
            hit_encontrado = False
            rank_hit = -1

            if modo_evaluacion == "human":
                # Modo auditoría humana: No evaluamos, solo preparamos la plantilla
                for rank, doc in enumerate(docs, start=1):
                    registro_human.append({
                        "query_id": query_id,
                        "pregunta": query,
                        "rank": rank,
                        "documento_recuperado": doc.page_content,
                        "texto_clave_esperado": texto_clave,
                        "calificacion_experto_1_o_0": ""
                    })
                continue # Saltamos métricas
                
            elif modo_evaluacion == "llm_judge":
                # Modo Juez LLM: Evaluación semántica
                llm = self._get_llm_judge()
                for rank, doc in enumerate(docs, start=1):
                    prompt = f"Eres un juez experto. Pregunta: '{query}'. Documento recuperado: '{doc.page_content}'. ¿El documento contiene suficiente información para responder o deducir la respuesta a la pregunta? Responde SOLO con el número 1 (si la responde) o 0 (si no la responde o es insuficiente)."
                    try:
                        respuesta_juez = llm.invoke(prompt).content.strip()
                        if "1" in respuesta_juez:
                            hit_encontrado = True
                            rank_hit = rank
                            break
                    except Exception as e:
                        print(f"[ERROR] Error invocando al Juez LLM: {e}")
            else:
                # Modo Default: Subcadena Exacta (Determinista)
                for rank, doc in enumerate(docs, start=1):
                    doc_norm = " ".join(doc.page_content.lower().split())
                    if texto_clave_norm in doc_norm:
                        hit_encontrado = True
                        rank_hit = rank
                        break
                        
            # 3. Calcular Métricas IR
            if modo_evaluacion in ["exact_match", "llm_judge"]:
                if hit_encontrado:
                    if rank_hit <= 5:
                        resultados_metricas["hits_at_5"] += 1
                    if rank_hit <= 10:
                        resultados_metricas["hits_at_10"] += 1
                        resultados_metricas["sum_map_10"] += (1.0 / rank_hit)
                        resultados_metricas["sum_ndcg_10"] += (1.0 / math.log2(rank_hit + 1))
                    
                    if verbose:
                        print(f"   ✅ HIT (Rank {rank_hit}) -> Query: {query_id}")
                else:
                    if verbose:
                        print(f"   ❌ MISS -> Query: {query_id}")

                registro_resultados.append({
                    "query_id": query_id,
                    "pregunta": query,
                    "estrategia": estrategia_nombre,
                    "hit": int(hit_encontrado),
                    "rank_encontrado": rank_hit if hit_encontrado else -1
                })

        # 4. Finalización y Exportación
        if modo_evaluacion == "human":
            df_human = pd.DataFrame(registro_human)
            try:
                archivo_excel = self.out_dir / f"auditoria_manual_{estrategia_nombre}.xlsx"
                # Instalar openpyxl silenciosamente si falta para excel
                df_human.to_excel(archivo_excel, index=False)
            except ImportError:
                print("⚠️ Falta 'openpyxl' para exportar a excel. Exportando a CSV en su lugar.")
                archivo_excel = self.out_dir / f"auditoria_manual_{estrategia_nombre}.csv"
                df_human.to_csv(archivo_excel, index=False, encoding='utf-8-sig')

            if verbose:
                print(f"\n=========================================")
                print(f"📝 MODO AUDITORÍA HUMANA COMPLETADO")
                print(f"Se exportaron {len(df_human)} fragmentos para evaluación experta.")
                print(f"Archivo generado: {archivo_excel}")
                print(f"=========================================\n")
            return df_human, None
            
        else:
            df_resultados = pd.DataFrame(registro_resultados)
            archivo_csv = self.out_dir / f"resultados_{modo_evaluacion}_{estrategia_nombre}.csv"
            df_resultados.to_csv(archivo_csv, index=False, encoding='utf-8-sig')
            
            n = resultados_metricas["total_queries"]
            avg_latencia = sum(latencias)/len(latencias) if latencias else 0.0
            avg_tokens = sum(tokens_contexto)/len(tokens_contexto) if tokens_contexto else 0.0
            
            metricas = {
                "estrategia": estrategia_nombre,
                "modo_evaluacion": modo_evaluacion,
                "Recall@5": round((resultados_metricas["hits_at_5"] / n) * 100, 2) if n > 0 else 0,
                "Recall@10": round((resultados_metricas["hits_at_10"] / n) * 100, 2) if n > 0 else 0,
                "MAP@10": round(resultados_metricas["sum_map_10"] / n, 4) if n > 0 else 0,
                "NDCG@10": round(resultados_metricas["sum_ndcg_10"] / n, 4) if n > 0 else 0,
                "Latencia_Promedio_Segundos": round(avg_latencia, 4),
                "Tokens_Contexto_Promedio": round(avg_tokens, 1)
            }
            
            if verbose:
                print(f"\n=========================================")
                print(f"🏆 RESULTADO: {estrategia_nombre} ({modo_evaluacion})")
                print(f" Recall@5:  {metricas['Recall@5']}%")
                print(f" Recall@10: {metricas['Recall@10']}%")
                print(f" MAP@10:    {metricas['MAP@10']}")
                print(f" NDCG@10:   {metricas['NDCG@10']}")
                print(f"=========================================\n")
                
            return df_resultados, metricas

    def calcular_metricas_desde_excel(self, ruta_excel: str | Path, estrategia_nombre: str = "Manual", verbose: bool = True):
        """
        Lee un archivo Excel previamente generado en modo 'human' y ya calificado por un experto.
        Calcula las métricas Recall@5, Recall@10, MAP@10 y NDCG@10 basándose en los '1's asignados.
        """
        ruta = Path(ruta_excel)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo de auditoría: {ruta}")
            
        try:
            if ruta.suffix == '.csv':
                df = pd.read_csv(ruta)
            else:
                df = pd.read_excel(ruta)
        except Exception as e:
            raise ValueError(f"Error al leer el archivo {ruta}: {e}")
            
        if 'calificacion_experto_1_o_0' not in df.columns:
            raise ValueError("El archivo no tiene la columna 'calificacion_experto_1_o_0'.")
            
        df['calificacion_experto_1_o_0'] = pd.to_numeric(df['calificacion_experto_1_o_0'], errors='coerce').fillna(0)
        
        grupos = df.groupby('query_id')
        
        n = len(self.ground_truth) # Total de queries en el dataset original
        hits_at_5 = 0
        hits_at_10 = 0
        sum_map_10 = 0.0
        sum_ndcg_10 = 0.0
        
        if verbose:
            print(f"📊 Procesando calificaciones manuales de '{estrategia_nombre}' desde '{ruta.name}'...")
        
        for query_id, grupo in grupos:
            hits_manuales = grupo[grupo['calificacion_experto_1_o_0'] == 1]
            if not hits_manuales.empty:
                # Tomar el rank del primer hit (el menor rank)
                rank_hit = hits_manuales['rank'].min()
                
                if rank_hit <= 5:
                    hits_at_5 += 1
                if rank_hit <= 10:
                    hits_at_10 += 1
                    sum_map_10 += (1.0 / rank_hit)
                    sum_ndcg_10 += (1.0 / math.log2(rank_hit + 1))
                    
        metricas = {
            "estrategia": estrategia_nombre,
            "modo_evaluacion": "human_reviewed",
            "Recall@5": round((hits_at_5 / n) * 100, 2) if n > 0 else 0,
            "Recall@10": round((hits_at_10 / n) * 100, 2) if n > 0 else 0,
            "MAP@10": round(sum_map_10 / n, 4) if n > 0 else 0,
            "NDCG@10": round(sum_ndcg_10 / n, 4) if n > 0 else 0,
            "Latencia_Promedio_Segundos": "N/A", # No hay latencia en revisión manual
            "Tokens_Contexto_Promedio": "N/A"
        }
        
        if verbose:
            print(f"\n=========================================")
            print(f"🏆 RESULTADO DE REVISIÓN MANUAL: {estrategia_nombre}")
            print(f" Consultas Evaluadas: {len(grupos)} / Total GT: {n}")
            print(f" Recall@5:  {metricas['Recall@5']}%")
            print(f" Recall@10: {metricas['Recall@10']}%")
            print(f" MAP@10:    {metricas['MAP@10']}")
            print(f" NDCG@10:   {metricas['NDCG@10']}")
            print(f"=========================================\n")
            
        return metricas
