import numpy as np

class RastreadorTelemetria:
    """
    Laboratorio de pruebas: Rastrea latencia y calcula Costo Total de Propiedad (TCO)
    para el análisis de Pareto.
    """
    
    # Precios por 1 Millón de tokens (estimados genéricos para el ejercicio)
    PRECIOS_LLM = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "llama3.1:8b-local": {"input": 0.00, "output": 0.00}, # El costo local se mide en HW/Electricidad, $0 API
        "hibrido": {"input": 0.00, "output": 0.00} # Retrieval puro
    }

    def __init__(self):
        self.registros = []

    def registrar_consulta(self, modelo: str, latencia_segundos: float, tokens_input: int, tokens_output: int):
        """Registra los datos operacionales de una consulta."""
        self.registros.append({
            "modelo": modelo,
            "latencia": latencia_segundos,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output
        })
        
    def calcular_metricas(self, modelo: str):
        """Calcula agregados (P50, P95, Costo) para un modelo específico."""
        datos_modelo = [r for r in self.registros if r["modelo"] == modelo]
        
        if not datos_modelo:
            return None
            
        latencias = [r["latencia"] for r in datos_modelo]
        total_in = sum(r["tokens_input"] for r in datos_modelo)
        total_out = sum(r["tokens_output"] for r in datos_modelo)
        
        # Calcular latencias P50 (mediana) y P95
        p50 = np.percentile(latencias, 50)
        p95 = np.percentile(latencias, 95)
        
        # Calcular costo (Regla de 3 para precio por millón)
        tarifas = self.PRECIOS_LLM.get(modelo, {"input": 0.0, "output": 0.0})
        costo_in = (total_in / 1_000_000) * tarifas["input"]
        costo_out = (total_out / 1_000_000) * tarifas["output"]
        costo_total_usd = costo_in + costo_out
        
        # Calcular costo promedio por cada 1000 consultas (Métrica solicitada en E6)
        n_consultas = len(datos_modelo)
        costo_per_1k = (costo_total_usd / n_consultas) * 1000 if n_consultas > 0 else 0
        
        return {
            "modelo": modelo,
            "muestras": n_consultas,
            "latencia_p50": p50,
            "latencia_p95": p95,
            "costo_total_usd": costo_total_usd,
            "costo_por_1000_consultas": costo_per_1k
        }
