"""
test_regresion_prompts.py

Demostración de CI/CD para LLMOps: Prevención de Regresiones.
Este script emula una barrera de seguridad automatizada. Si un desarrollador
intenta subir un cambio de prompt que degrada el Faithfulness (causando alucinaciones),
el pipeline falla y bloquea el despliegue a producción.
"""
import time
import sys

def color_print(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")

def run_golden_dataset_eval(prompt_version: str, is_bad_prompt: bool = False):
    """
    Simula la ejecución del LLM-as-a-Judge sobre un Golden Dataset de 30 preguntas.
    Retorna el score promedio de Faithfulness (Fidelidad a la Normativa).
    """
    print(f"🔄 Corriendo suite de evaluaciones sobre Golden Dataset (n=30) usando '{prompt_version}'...")
    time.sleep(1.5) # Simular tiempo de inferencia
    
    if is_bad_prompt:
        # Un prompt malo genera alucinaciones, bajando el score
        return 0.68 
    else:
        # El prompt de producción actual es altamente preciso
        return 0.94 

def main():
    color_print("=== 🚀 INICIANDO PIPELINE CI/CD (LLMOps) ===", "1;34")
    
    # 1. Obtener la métrica base (Producción actual)
    color_print("\n[Paso 1] Evaluando Baseline en Producción...", "1;33")
    baseline_score = run_golden_dataset_eval("Prompt v1.0 (Producción)", is_bad_prompt=False)
    print(f"📊 Faithfulness Baseline: {baseline_score:.2f}")
    
    # 2. Evaluar el nuevo código/prompt propuesto por el desarrollador
    color_print("\n[Paso 2] Evaluando Nuevo Commit...", "1;33")
    # Simulamos que el desarrollador hizo un cambio que rompió el contexto estricto
    new_score = run_golden_dataset_eval("Prompt v1.1 (Commit Propuesto)", is_bad_prompt=True)
    print(f"📊 Faithfulness Nuevo Commit: {new_score:.2f}")
    
    # 3. Gatekeeper (Puerta de Paso)
    color_print("\n[Paso 3] Análisis de Regresión...", "1;33")
    delta = new_score - baseline_score
    
    print(f"📉 Diferencia de rendimiento: {delta:.2f}")
    
    if new_score < 0.85 or delta < -0.05:
        color_print("\n❌ ALERTA: REGRESIÓN DETECTADA.", "1;31")
        color_print("El nuevo prompt reduce la precisión normativa por debajo del umbral aceptable.", "1;31")
        color_print("⛔ ACCIÓN: Bloqueando despliegue a producción (Exit Code 1).", "1;31")
        sys.exit(1)
    else:
        color_print("\n✅ VERIFICACIÓN EXITOSA.", "1;32")
        color_print("El nuevo prompt mantiene o mejora las métricas. Aprobando despliegue.", "1;32")
        sys.exit(0)

if __name__ == "__main__":
    main()
