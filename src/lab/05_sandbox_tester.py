import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.append(project_root)

from src.lab.telemetria import RastreadorTelemetria
from src.lab.graficos import plot_frontera_pareto

def main():
    print("================================================================================")
    print("🔬 SANDBOX TESTER: TELEMETRÍA Y PARETO (LAB)")
    print("================================================================================")
    
    # 1. Simular telemetría operacional para 4 modelos
    print("1️⃣ Simulando telemetría de 100 consultas para 4 modelos...")
    tracker = RastreadorTelemetria()
    
    # Mock data para Llama3 Local (Gratis pero medio lento, precisión decente)
    for _ in range(100):
        tracker.registrar_consulta("llama3.1:8b-local", latencia_segundos=2.5, tokens_input=5000, tokens_output=150)
        
    # Mock data para GPT-4o-mini (Barato y rapidísimo)
    for _ in range(100):
        tracker.registrar_consulta("gpt-4o-mini", latencia_segundos=0.8, tokens_input=5000, tokens_output=150)
        
    # Mock data para GPT-4o (Caro y moderado)
    for _ in range(100):
        tracker.registrar_consulta("gpt-4o", latencia_segundos=1.5, tokens_input=5000, tokens_output=150)
        
    # Mock data para Búsqueda Híbrida pura sin generador (Rapidísimo y Gratis)
    for _ in range(100):
        tracker.registrar_consulta("hibrido", latencia_segundos=0.3, tokens_input=0, tokens_output=0)
        
    # 2. Calcular agregados de telemetría y costo
    print("\n2️⃣ Calculando métricas de latencia P50/P95 y Costo Total de Propiedad (TCO)...")
    modelos = ["llama3.1:8b-local", "gpt-4o-mini", "gpt-4o", "hibrido"]
    
    # Simulamos que el NDCG ya lo calculó el evaluador
    # Estos números simulan el accuracy que sacaría cada uno
    mock_ndcg = {
        "hibrido": 0.65,              # Baseline léxico-semántico
        "llama3.1:8b-local": 0.72,    # Juez local
        "gpt-4o-mini": 0.81,          # Juez nube barato
        "gpt-4o": 0.88                # SOTA carísimo
    }
    
    resultados_grafico = []
    
    for mod in modelos:
        metricas = tracker.calcular_metricas(mod)
        if metricas:
            costo_1k = metricas["costo_por_1000_consultas"]
            lat_95 = metricas["latencia_p95"]
            ndcg_final = mock_ndcg[mod]
            
            print(f"  📊 {mod.upper()}:")
            print(f"     - Latencia P95: {lat_95:.2f} s")
            print(f"     - Costo (por 1K): ${costo_1k:.4f} USD")
            print(f"     - NDCG@10 Simulado: {ndcg_final}")
            
            # Formatear dict para la gráfica
            resultados_grafico.append({
                "modelo": mod,
                "costo_por_1000": costo_1k,
                "ndcg": ndcg_final
            })
            
    # 3. Dibujar la frontera de Pareto
    print("\n3️⃣ Generando el Gráfico de la Frontera de Pareto (Cost vs Accuracy)...")
    output_png = os.path.join(project_root, "data", "03_output", "pareto_sandbox_test.png")
    
    plot_frontera_pareto(resultados_grafico, output_png)
    print("\n✅ Sandbox Tester finalizado sin errores. Los módulos del Lab están listos para el Notebook Final.")

if __name__ == "__main__":
    main()
