import json
import math
import time
import random
import tiktoken
import pandas as pd
from pathlib import Path

class EvaluadorRAG:
    """
    Clase dedicada a realizar pruebas cuantitativas y análisis estadísticos 
    avanzados sobre el pipeline RAG contra un Dataset de Evaluación (Ground Truth).
    """
    def __init__(self, ground_truth_path: str | Path, subcarpeta_salida: str = "oficiales"):
        self.gt_path = Path(ground_truth_path)
        if not self.gt_path.exists():
            raise FileNotFoundError(f"No se encontró el Ground Truth en {self.gt_path}")
            
        with open(self.gt_path, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
            
        from datetime import datetime
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determinar el root del proyecto para guardar resultados
        self.project_root = self.gt_path.parent.parent
        self.out_dir = self.project_root / "data" / "03_output" / "evaluaciones" / subcarpeta_salida / f"run_{self.run_timestamp}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self._llm_judge = None

    def _get_llm_judge(self):
        if self._llm_judge is None:
            from src.nlp_core.config_llm import get_langchain_chat
            self._llm_judge = get_langchain_chat(task="qa", temperature=0.0)
        return self._llm_judge

    def calcular_bootstrap_ci(self, scores: list, num_resamples: int = 1000, confidence_level: float = 0.95) -> tuple[float, float, float]:
        """
        Calcula el intervalo de confianza (CI) para una lista de métricas individuales por consulta
        utilizando remuestreo Bootstrap no paramétrico.
        
        Retorna:
        - (media_original, limite_inferior, limite_superior)
        """
        if not scores:
            return 0.0, 0.0, 0.0
            
        random.seed(42) # Asegurar reproducibilidad científica de la evaluación
        
        n = len(scores)
        medias_bootstrap = []
        
        for _ in range(num_resamples):
            muestra_resampled = random.choices(scores, k=n)
            medias_bootstrap.append(sum(muestra_resampled) / n)
            
        medias_bootstrap.sort()
        
        alfa = 1.0 - confidence_level
        idx_inf = int((alfa / 2.0) * num_resamples)
        idx_sup = int((1.0 - (alfa / 2.0)) * num_resamples) - 1
        
        media_original = sum(scores) / n
        lim_inf = medias_bootstrap[max(0, idx_inf)]
        lim_sup = medias_bootstrap[min(num_resamples - 1, idx_sup)]
        
        return round(media_original, 4), round(lim_inf, 4), round(lim_sup, 4)

    def evaluar_retrieval(self, funcion_busqueda, estrategia_nombre: str, k: int = 10, verbose: bool = True, modo_evaluacion: str = "exact_match", limite_consultas: int = None, metadatos_llm: dict = None):
        """
        Evalúa una función de búsqueda específica contra el Ground Truth usando métricas de IR
        e incorpora Intervalos de Confianza Bootstrap del 95% (MA5).
        Acepta metadatos_llm (dict) para guardar configuraciones del modelo.
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
        
        registro_resultados = [] 
        latencias = []
        tokens_contexto = []
        
        # Listas para almacenar scores individuales por consulta (Requerido para Bootstrap)
        ndcg_scores = []
        map_scores = []
        recall_scores = []
        
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

            # 2. Evaluación del Hit
            hit_encontrado = False
            rank_hit = -1

            if modo_evaluacion == "human":
                # Guardamos para la plantilla
                for rank, doc in enumerate(docs, start=1):
                    registro_resultados.append({
                        "query_id": query_id,
                        "pregunta": query,
                        "rank": rank,
                        "documento_recuperado": doc.page_content,
                        "texto_clave_esperado": texto_clave,
                        "calificacion_experto_1_o_0": ""
                    })
                continue
                
            elif modo_evaluacion == "llm_judge":
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
                # exact_match
                for rank, doc in enumerate(docs, start=1):
                    doc_norm = " ".join(doc.page_content.lower().split())
                    if texto_clave_norm in doc_norm:
                        hit_encontrado = True
                        rank_hit = rank
                        break
                        
            # 3. Calcular Métricas por Consulta
            ndcg_val = 0.0
            map_val = 0.0
            recall_val = 0
            
            if hit_encontrado:
                if rank_hit <= 5:
                    resultados_metricas["hits_at_5"] += 1
                if rank_hit <= 10:
                    resultados_metricas["hits_at_10"] += 1
                    map_val = 1.0 / rank_hit
                    ndcg_val = 1.0 / math.log2(rank_hit + 1)
                    recall_val = 1
                    
                    resultados_metricas["sum_map_10"] += map_val
                    resultados_metricas["sum_ndcg_10"] += ndcg_val
                
                if verbose:
                    print(f"   ✅ HIT (Rank {rank_hit}) -> Query: {query_id}")
            else:
                if verbose:
                    print(f"   ❌ MISS -> Query: {query_id}")

            ndcg_scores.append(ndcg_val)
            map_scores.append(map_val)
            recall_scores.append(recall_val)

            registro_resultados.append({
                "query_id": query_id,
                "pregunta": query,
                "estrategia": estrategia_nombre,
                "hit": int(hit_encontrado),
                "rank_encontrado": rank_hit if hit_encontrado else -1,
                "ndcg_10": ndcg_val,
                "map_10": map_val
            })

        # 4. Finalización y Exportación
        if modo_evaluacion == "human":
            df_human = pd.DataFrame(registro_resultados)
            try:
                archivo_excel = self.out_dir / f"auditoria_manual_{estrategia_nombre}.xlsx"
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
            archivo_csv = self.out_dir / f"resultados_{modo_evaluacion}_{estrategia_nombre}_{self.run_timestamp}.csv"
            df_resultados.to_csv(archivo_csv, index=False, encoding='utf-8-sig')
            
            n = resultados_metricas["total_queries"]
            import numpy as np
            p50 = np.percentile(latencias, 50) if latencias else 0.0
            p95 = np.percentile(latencias, 95) if latencias else 0.0
            p99 = np.percentile(latencias, 99) if latencias else 0.0
            avg_tokens = sum(tokens_contexto)/len(tokens_contexto) if tokens_contexto else 0.0
            
            # Calcular CIs Bootstrap
            ndcg_avg, ndcg_inf, ndcg_sup = self.calcular_bootstrap_ci(ndcg_scores)
            recall_avg, recall_inf, recall_sup = self.calcular_bootstrap_ci(recall_scores)
            map_avg, map_inf, map_sup = self.calcular_bootstrap_ci(map_scores)
            
            # Estimar Costo Operativo
            costo_total_usd = 0.0
            if metadatos_llm and not metadatos_llm.get("Es_QA_Local", True):
                modelo_qa = metadatos_llm.get("LLM_Modelo_QA", "gpt-4o-mini")
                # Precios estándar por 1 Millón de tokens (estimado para Arena)
                precio_in = 0.15 if "mini" in modelo_qa else 5.0
                precio_out = 0.60 if "mini" in modelo_qa else 15.0
                
                # Para la Arena (evaluación de retrieval), estimamos el costo de lo que
                # costaría enviar este contexto (avg_tokens) + generar una respuesta corta (~300 tokens)
                costo_por_consulta = (avg_tokens / 1_000_000) * precio_in + (300 / 1_000_000) * precio_out
                costo_total_usd = costo_por_consulta * 1000 # Costo proyectado por cada 1,000 consultas
            
            metricas = {
                "estrategia": estrategia_nombre,
                "modo_evaluacion": modo_evaluacion,
                "Recall@5": round((resultados_metricas["hits_at_5"] / n) * 100, 2) if n > 0 else 0,
                "Recall@10": round((resultados_metricas["hits_at_10"] / n) * 100, 2) if n > 0 else 0,
                "Recall@10_CI_95": (round(recall_inf * 100, 2), round(recall_sup * 100, 2)),
                "MAP@10": round(resultados_metricas["sum_map_10"] / n, 4) if n > 0 else 0,
                "MAP@10_CI_95": (round(map_inf, 4), round(map_sup, 4)),
                "NDCG@10": round(resultados_metricas["sum_ndcg_10"] / n, 4) if n > 0 else 0,
                "NDCG@10_CI_95": (round(ndcg_inf, 4), round(ndcg_sup, 4)),
                "Latencia_P50": round(p50, 4),
                "Latencia_P95": round(p95, 4),
                "Latencia_P99": round(p99, 4),
                "Tokens_Contexto_Promedio": round(avg_tokens, 1),
                "Costo_Total_USD": round(costo_total_usd, 4)
            }
            
            # Integrar metadatos (Modelos, Local vs Nube)
            if metadatos_llm:
                metricas.update(metadatos_llm)
            
            if verbose:
                print(f"\n=========================================")
                print(f"🏆 RESULTADO: {estrategia_nombre} ({modo_evaluacion})")
                print(f" Recall@5:  {metricas['Recall@5']}%")
                print(f" Recall@10: {metricas['Recall@10']}% | CI 95%: [{metricas['Recall@10_CI_95'][0]}% - {metricas['Recall@10_CI_95'][1]}%]")
                print(f" MAP@10:    {metricas['MAP@10']} | CI 95%: [{metricas['MAP@10_CI_95'][0]} - {metricas['MAP@10_CI_95'][1]}]")
                print(f" NDCG@10:   {metricas['NDCG@10']} | CI 95%: [{metricas['NDCG@10_CI_95'][0]} - {metricas['NDCG@10_CI_95'][1]}]")
                print(f"=========================================\n")
                
            return df_resultados, metricas

    def evaluar_contaminacion_ciega(self, candidato_nombre: str, task: str = "qa", limite_consultas: int = None, verbose: bool = True, metadatos_llm: dict = None):
        """
        Ejecuta una evaluación a ciegas (Data Contamination Check)
        donde el LLM responde directamente de su memoria sin RAG.
        """
        if verbose:
            print(f"🧪 Iniciando Data Contamination Check (Evaluación Ciega) para '{candidato_nombre}'...")
            
        from src.nlp_core.config_llm import get_langchain_chat
        
        llm_candidato = get_langchain_chat(task=task, temperature=0.0)
        llm_juez = self._get_llm_judge()
        
        consultas_a_evaluar = self.ground_truth
        if limite_consultas is not None:
            consultas_a_evaluar = consultas_a_evaluar[:limite_consultas]
            
        hits = 0
        scores_individuales = []
        registro_detallado = []
        
        for item in consultas_a_evaluar:
            query = item['pregunta']
            query_id = item.get('query_id', 'N/A')
            doc_esperado = item.get('documentos_esperados', [{}])[0]
            texto_clave = doc_esperado.get('texto_clave_esperado', '')
            
            if not texto_clave:
                continue
                
            # 1. Inferencia Ciega
            prompt_ciego = (
                f"Responde a la siguiente pregunta técnica sobre regulación financiera bancaria utilizando "
                f"exclusivamente tus conocimientos pre-entrenados de forma precisa:\n\nPregunta: {query}"
            )
            
            t_inicio = time.time()
            try:
                respuesta = llm_candidato.invoke(prompt_ciego).content.strip()
                duracion = time.time() - t_inicio
            except Exception as e:
                print(f"[ERROR] Error al invocar LLM candidato '{candidato_nombre}' en consulta {query_id}: {e}")
                continue
                
            # 2. Evaluar con Juez LLM
            prompt_juez = (
                f"Eres un juez experto en regulación financiera. Evaluamos si el modelo respondió correctamente de memoria.\n"
                f"Pregunta: '{query}'\n"
                f"Respuesta del modelo: '{respuesta}'\n"
                f"Texto de referencia correcto esperado: '{texto_clave}'\n\n"
                f"¿La respuesta del modelo es correcta, precisa y coincide con la información del texto de referencia? "
                f"Responde ÚNICAMENTE con el dígito 1 (si es correcta) o 0 (si es incorrecta, insuficiente o alucinada)."
            )
            
            hit = 0
            try:
                fallo_juez = llm_juez.invoke(prompt_juez).content.strip()
                if "1" in fallo_juez:
                    hit = 1
                    hits += 1
            except Exception as e:
                print(f"[ERROR] Error invocando al Juez LLM: {e}")
                
            scores_individuales.append(hit)
            
            if verbose:
                estado = "✅ CORRECTO (Contaminación)" if hit == 1 else "❌ INCORRECTO (Limpio)"
                print(f"   Query {query_id}: {estado}")
                
            registro_detallado.append({
                "query_id": query_id,
                "pregunta": query,
                "respuesta_ciega": respuesta,
                "texto_clave_esperado": texto_clave,
                "hit_ciego": hit,
                "duracion_segundos": round(duracion, 4)
            })
            
        n = len(scores_individuales)
        precision_ciega = (hits / n) * 100 if n > 0 else 0.0
        
        # Calcular CIs Bootstrap
        _, lim_inf, lim_sup = self.calcular_bootstrap_ci(scores_individuales)
        
        resultados = {
            "candidato": candidato_nombre,
            "queries_evaluadas": n,
            "Precision_Ciega_Porcentaje": round(precision_ciega, 2),
            "CI_95_Inferior": round(lim_inf * 100, 2),
            "CI_95_Superior": round(lim_sup * 100, 2)
        }
        
        if metadatos_llm:
            resultados.update(metadatos_llm)
        
        # Guardar en archivo CSV
        df = pd.DataFrame(registro_detallado)
        candidato_seguro = candidato_nombre.replace(":", "-")
        archivo_out = self.out_dir / f"contaminacion_ciega_{candidato_seguro}_{self.run_timestamp}.csv"
        df.to_csv(archivo_out, index=False, encoding='utf-8-sig')
        
        if verbose:
            print("\n=========================================")
            print(f"📊 RESULTADO DE CONTAMINACIÓN: {candidato_nombre}")
            print(f" Precisión de Memoria: {resultados['Precision_Ciega_Porcentaje']}%")
            print(f" Intervalo Confianza 95%: [{resultados['CI_95_Inferior']}% - {resultados['CI_95_Superior']}%]")
            print(f" Archivo guardado: {archivo_out}")
            print("=========================================\n")
            
        return resultados

    def evaluar_desagregacion_errores(self, funcion_busqueda, funcion_qa_extraccion, estrategia_nombre: str, limite_consultas: int = None, verbose: bool = True, metadatos_llm: dict = None, tarea: str = "extraccion"):
        """
        Realiza una evaluación completa RAG y clasifica los fallos
        en tres categorías mutuamente excluyentes:
        - A: Fallo de Retrieval (texto clave no en Top-K)
        - B: Alucinación/Fallo de LLM (contexto correcto, pero respuesta incorrecta)
        - C: Fallo de Formato/Parser (respuesta semántica correcta, pero falló validación estructurada)
        """
        if verbose:
            print(f"🔍 Iniciando Análisis de Errores Desagregado por Etapas para '{estrategia_nombre}'...")
            
        llm_juez = self._get_llm_judge()
        
        consultas_a_evaluar = self.ground_truth
        if limite_consultas is not None:
            consultas_a_evaluar = consultas_a_evaluar[:limite_consultas]
            
        conteo_errores = {
            "Total_Consultas": len(consultas_a_evaluar),
            "Exitosos": 0,
            "Fallo_Retrieval_A": 0,
            "Fallo_Generacion_B": 0,
            "Fallo_Formato_C": 0
        }
        
        registro_desagregado = []
        
        total_queries = len(consultas_a_evaluar)
        for idx, item in enumerate(consultas_a_evaluar, 1):
            print(f"⌛ [Progreso Fase 3] Evaluando consulta {idx}/{total_queries}...")
            query = item['pregunta']
            query_id = item.get('query_id', 'N/A')
            doc_esperado = item.get('documentos_esperados', [{}])[0]
            texto_clave = doc_esperado.get('texto_clave_esperado', '')
            
            if not texto_clave:
                continue
                
            texto_clave_norm = " ".join(texto_clave.lower().split())
            
            # --- ETAPA 1: BÚSQUEDA (RETRIEVAL) ---
            docs_retrieved = []
            retrieval_ok = False
            
            try:
                # Buscamos en Top-10 para una evaluación robusta
                resultados = funcion_busqueda(query, k=10)
                if isinstance(resultados, tuple) and len(resultados) == 2:
                    docs_retrieved = resultados[1]
                else:
                    docs_retrieved = resultados
            except Exception as e:
                print(f"[ERROR] Error al buscar en query {query_id}: {e}")
                conteo_errores["Fallo_Retrieval_A"] += 1
                registro_desagregado.append({
                    "query_id": query_id,
                    "categoria_error": "A (Retrieval - Excepción)",
                    "detalle": str(e)
                })
                continue
                
            # Verificar si el texto clave está en los fragmentos
            for doc in docs_retrieved:
                doc_norm = " ".join(doc.page_content.lower().split())
                if texto_clave_norm in doc_norm:
                    retrieval_ok = True
                    break
                    
            # --- ETAPA 2: GENERACIÓN Y PARSEO ---
            generacion_ok = False
            formato_ok = True
            error_formato_detalle = ""
            respuesta_texto = ""
            prompt_hash_extraido = "N/A"
            prompt_version_extraido = "N/A"
            modelo_usado = "N/A"
            
            try:
                # La función debe ejecutar el extractor Pydantic/estructurado
                resultado_extraccion, info = funcion_qa_extraccion(query, k=10)
                
                if isinstance(info, dict):
                    prompt_hash_extraido = info.get("prompt_hash", "N/A")
                    prompt_version_extraido = info.get("prompt_version", "N/A")
                    modelo_usado = info.get("modelo", "N/A")
                
                # Serializar el resultado para el juez
                if hasattr(resultado_extraccion, "model_dump_json"):
                    respuesta_texto = resultado_extraccion.model_dump_json()
                elif hasattr(resultado_extraccion, "dict"):
                    respuesta_texto = json.dumps(resultado_extraccion.dict())
                else:
                    respuesta_texto = str(resultado_extraccion)
                    
            except (json.JSONDecodeError, ValueError) as fe:
                formato_ok = False
                error_formato_detalle = f"JSON/Value Error: {str(fe)}"
            except Exception as e:
                # Pydantic validation error or general parser error
                if "ValidationError" in type(e).__name__:
                    formato_ok = False
                    error_formato_detalle = f"Pydantic Validation Error: {str(e)}"
                else:
                    formato_ok = False
                    error_formato_detalle = f"General Ingestion Error: {str(e)}"
                    
            if tarea == "qa":
                formato_ok = True # En QA no hay errores de formato Pydantic
                error_formato_detalle = ""
                    
            # Evaluar veracidad semántica si no falló el parser
            if formato_ok:
                if tarea == "extraccion":
                    prompt_juez = (
                        f"Eres un juez experto en validación de extracción de datos regulatorios.\n"
                        f"Pregunta del analista: '{query}'\n"
                        f"JSON Extraído por el modelo: '{respuesta_texto}'\n"
                        f"Texto de referencia correcto esperado (Ground Truth): '{texto_clave}'\n\n"
                        f"¿El JSON extraído contiene de forma exacta, correcta y sin alucinaciones "
                        f"la información del texto de referencia correcto? "
                        f"Responde ÚNICAMENTE con el dígito 1 (si es correcta y verídica) o 0 (si es incorrecta, omite campos críticos o tiene alucinaciones)."
                    )
                else:
                    prompt_juez = (
                        f"Eres un juez experto en regulación financiera.\n"
                        f"Pregunta del analista: '{query}'\n"
                        f"Respuesta del modelo: '{respuesta_texto}'\n"
                        f"Texto de referencia correcto esperado (Ground Truth): '{texto_clave}'\n\n"
                        f"¿La respuesta del modelo contiene de forma exacta, correcta y sin alucinaciones "
                        f"la información del texto de referencia correcto? "
                        f"Responde ÚNICAMENTE con el dígito 1 (si es correcta y verídica) o 0 (si es incorrecta, omite información crítica o tiene alucinaciones)."
                    )
                try:
                    respuesta_juez = llm_juez.invoke(prompt_juez).content.strip()
                    if "1" in respuesta_juez:
                        generacion_ok = True
                except Exception as e:
                    print(f"[ERROR] Error al calificar con Juez LLM en query {query_id}: {e}")
                    
            # --- CLASIFICACIÓN FINAL ---
            categoria_final = "ÉXITO"
            detalle_final = "Extracción e inyección correctas."
            
            if retrieval_ok and formato_ok and generacion_ok:
                conteo_errores["Exitosos"] += 1
            else:
                if not retrieval_ok:
                    categoria_final = "A"
                    detalle_final = "El buscador falló. El texto clave no estuvo presente en los chunks recuperados en el Top-10."
                    conteo_errores["Fallo_Retrieval_A"] += 1
                elif not formato_ok:
                    categoria_final = "C"
                    detalle_final = f"Fallo de Formato/Parser Pydantic: {error_formato_detalle}"
                    conteo_errores["Fallo_Formato_C"] += 1
                else:
                    categoria_final = "B"
                    detalle_final = "Alucinación o Inexactitud del LLM. El contexto contenía el fragmento pero el LLM omitió variables o alucinó datos."
                    conteo_errores["Fallo_Generacion_B"] += 1
                    
            if verbose:
                print(f"   Query {query_id}: {categoria_final} -> {detalle_final[:80]}...")
                
            registro_desagregado.append({
                "query_id": query_id,
                "pregunta": query,
                "retrieval_exitoso": int(retrieval_ok),
                "formato_exitoso": int(formato_ok),
                "generacion_exitosa": int(generacion_ok),
                "categoria_error": categoria_final,
                "detalle_error": detalle_final,
                "texto_esperado": texto_clave,
                "respuesta_generada": respuesta_texto,
                "modelo_extraccion_usado": modelo_usado,
                "prompt_version": prompt_version_extraido,
                "prompt_hash": prompt_hash_extraido
            })
            
        # Generar Reporte CSV
        df = pd.DataFrame(registro_desagregado)
        archivo_out = self.out_dir / f"analisis_errores_desagregados_{estrategia_nombre}_{self.run_timestamp}.csv"
        df.to_csv(archivo_out, index=False, encoding='utf-8-sig')
        
        if metadatos_llm:
            conteo_errores.update(metadatos_llm)
        
        # Calcular porcentajes
        total_errores = conteo_errores["Fallo_Retrieval_A"] + conteo_errores["Fallo_Generacion_B"] + conteo_errores["Fallo_Formato_C"]
        porc_retrieval = (conteo_errores["Fallo_Retrieval_A"] / total_errores) * 100 if total_errores > 0 else 0.0
        porc_generacion = (conteo_errores["Fallo_Generacion_B"] / total_errores) * 100 if total_errores > 0 else 0.0
        porc_formato = (conteo_errores["Fallo_Formato_C"] / total_errores) * 100 if total_errores > 0 else 0.0
        
        if verbose:
            print("\n=========================================")
            print(f"📊 REPORTE DE ERRORES DESAGREGADOS: {estrategia_nombre}")
            print(f" Total de consultas: {conteo_errores['Total_Consultas']}")
            print(f" Exitosas:           {conteo_errores['Exitosos']}")
            print(f" Total Fallas:       {total_errores}")
            print(f"   ↳ Fallo de Retrieval (A):      {conteo_errores['Fallo_Retrieval_A']} ({porc_retrieval:.1f}%)")
            print(f"   ↳ Fallo de Generación/Aluc (B): {conteo_errores['Fallo_Generacion_B']} ({porc_generacion:.1f}%)")
            print(f"   ↳ Fallo de Formato/Parser (C):  {conteo_errores['Fallo_Formato_C']} ({porc_formato:.1f}%)")
            print(f" Archivo guardado: {archivo_out}")
            print("=========================================\n")
            
        return conteo_errores, registro_desagregado

    def calcular_metricas_desde_excel(self, ruta_excel: str | Path, estrategia_nombre: str = "Manual", verbose: bool = True):
        """
        Lee un archivo Excel calificado por un experto y calcula las métricas.
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
        
        n = len(self.ground_truth) 
        hits_at_5 = 0
        hits_at_10 = 0
        sum_map_10 = 0.0
        sum_ndcg_10 = 0.0
        
        if verbose:
            print(f"📊 Procesando calificaciones manuales de '{estrategia_nombre}' desde '{ruta.name}'...")
        
        for query_id, grupo in grupos:
            hits_manuales = grupo[grupo['calificacion_experto_1_o_0'] == 1]
            if not hits_manuales.empty:
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
            "Latencia_Promedio_Segundos": "N/A",
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

def evaluar_faithfulness_claims(respuesta: str, contexto: str, llm_judge=None) -> float:
    """
    Evalúa qué tan fiel (faithful) es una respuesta generada basándose estrictamente 
    en el contexto proporcionado, utilizando la descomposición en Claims Atómicos.
    Retorna un score entre 0.0 y 1.0.
    """
    if not respuesta or not contexto:
        return 0.0
        
    if llm_judge is None:
        from src.nlp_core.config_llm import get_langchain_chat
        llm_judge = get_langchain_chat(task="judge", temperature=0.0)
        
    # Paso 1: Extraer claims
    prompt_extraccion = (
        f"Dada la siguiente respuesta, divídela en declaraciones (claims) individuales, atómicas e independientes.\n"
        f"Devuelve CADA CLAIM EN UNA LÍNEA NUEVA comenzando con un guion (-).\n\n"
        f"Respuesta:\n{respuesta}"
    )
    
    try:
        resultado_claims = llm_judge.invoke(prompt_extraccion).content.strip()
        claims = [line.strip().lstrip('-').strip() for line in resultado_claims.split('\n') if line.strip().startswith('-')]
    except Exception as e:
        print(f"[FAITHFULNESS ERROR] Extracción de claims falló: {e}")
        return 0.0
        
    if not claims:
        return 0.0
        
    # Paso 2: Verificar cada claim
    claims_soportados = 0
    for claim in claims:
        prompt_verificacion = (
            f"Contexto de Referencia:\n{contexto}\n\n"
            f"Declaración (Claim): '{claim}'\n\n"
            f"¿La declaración está completamente soportada (backed up) por el contexto de referencia?\n"
            f"Responde ÚNICAMENTE con 1 (Sí) o 0 (No)."
        )
        try:
            verificacion = llm_judge.invoke(prompt_verificacion).content.strip()
            if "1" in verificacion:
                claims_soportados += 1
        except Exception:
            pass
            
    # Paso 3: Calcular score
    return claims_soportados / len(claims)

def evaluar_answer_relevance(pregunta: str, respuesta: str, embeddings=None, n_preguntas: int = 3) -> float:
    """
    Calcula el Answer Relevance generando N preguntas a partir de la respuesta
    y comparando la similitud coseno de sus embeddings contra la pregunta original.
    """
    if not pregunta or not respuesta:
        return 0.0
        
    from src.nlp_core.config_llm import get_langchain_chat, get_embeddings
    import numpy as np
    from numpy.linalg import norm
    
    llm = get_langchain_chat(task="judge", temperature=0.0)
    if embeddings is None:
        embeddings = get_embeddings()
        
    prompt = (
        f"Dada la siguiente respuesta, genera {n_preguntas} posibles preguntas cortas "
        f"para las cuales esta respuesta sería la solución perfecta.\n"
        f"Devuelve UNA PREGUNTA POR LÍNEA comenzando con un guion (-).\n\n"
        f"Respuesta:\n{respuesta}"
    )
    
    try:
        preguntas_generadas_raw = llm.invoke(prompt).content.strip()
        preguntas_generadas = [line.strip().lstrip('-').strip() for line in preguntas_generadas_raw.split('\n') if line.strip().startswith('-')]
        preguntas_generadas = preguntas_generadas[:n_preguntas]
    except Exception as e:
        print(f"[RELEVANCE ERROR] Generación de preguntas falló: {e}")
        return 0.0
        
    if not preguntas_generadas:
        return 0.0
        
    try:
        emb_original = embeddings.embed_query(pregunta)
        emb_generadas = embeddings.embed_documents(preguntas_generadas)
        
        similitudes = []
        for emb_g in emb_generadas:
            cosine_sim = np.dot(emb_original, emb_g) / (norm(emb_original) * norm(emb_g))
            similitudes.append(cosine_sim)
            
        return sum(similitudes) / len(similitudes)
    except Exception as e:
        print(f"[RELEVANCE ERROR] Cálculo de embeddings falló: {e}")
        return 0.0

def evaluar_context_relevancy(pregunta: str, contexto: str, llm_judge=None) -> float:
    """
    Evalúa qué tan relevante es el contexto recuperado (Context Relevancy de RAGAS sin referencia).
    Calcula la proporción de oraciones del contexto que son útiles para responder la pregunta.
    """
    if not pregunta or not contexto:
        return 0.0
        
    if llm_judge is None:
        from src.nlp_core.config_llm import get_langchain_chat
        llm_judge = get_langchain_chat(task="judge", temperature=0.0)
        
    prompt = (
        f"Dada la siguiente pregunta y el contexto de búsqueda, extrae estrictamente "
        f"las oraciones del contexto que son directamente útiles para responder a la pregunta.\n"
        f"Si ninguna oración es útil, responde 'Ninguna'.\n"
        f"Devuelve cada oración útil en una nueva línea con un guion (-).\n\n"
        f"Pregunta: {pregunta}\n"
        f"Contexto:\n{contexto}"
    )
    
    try:
        resultado = llm_judge.invoke(prompt).content.strip()
        if "Ninguna" in resultado or not resultado:
            return 0.0
            
        oraciones_utiles = [line for line in resultado.split('\n') if line.strip().startswith('-')]
        
        # Una estimación rápida de oraciones totales en el contexto (por punto y seguido)
        oraciones_totales = max(1, len(contexto.split('. ')))
        
        score = len(oraciones_utiles) / oraciones_totales
        return min(1.0, score) # Asegurar límite de 1.0
    except Exception as e:
        print(f"[CONTEXT RELEVANCY ERROR] {e}")
        return 0.0


