import os

filepath = 'docs/entregas/Avance_5_Reporte_Final_Evaluaciones.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

header_marker = "---"

# Find the second occurrence of '---' which is after the title
first_dash = content.find(header_marker)
second_dash = content.find(header_marker, first_dash + 1)
third_dash = content.find(header_marker, second_dash + 1)

contexto = """

## 0. Contexto de Partida (Herencia del Avance 4)

En la fase de evaluación anterior consolidamos un marco MLOps riguroso que expuso hallazgos críticos para el Banco de México:
1. **Leniency Bias y Contaminación:** Los modelos locales (ej. Llama 3.1 8B) demostraron ser excelentes motores de recuperación a costo $0, pero fallan como evaluadores (aprobando el 37.6% de respuestas ciegas inventadas).
2. **Arquitectura Model Cascading:** Esta evidencia justificó migrar a una arquitectura híbrida: delegar el volumen de búsquedas (Retrieval/Reranking) al motor local, y reservar la nube (`gpt-4o`) exclusivamente para tareas de evaluación (LLM-as-a-Judge) y estructuración JSON compleja.
3. **Telemetría Operativa:** Implementamos tracking de latencia y consumo de tokens (registrado en `.jsonl` y visualizado en Streamlit) para monitorear el TCO del esquema híbrido.

**Objetivo del Avance 5:** Tomando esta arquitectura *Model Cascading* como línea base, el objetivo ahora es validar empírica y matemáticamente la sinergia de este ensamble (Diversidad y Consistencia) y auditar la resiliencia del modelo frente a ataques adversarios (Red-Teaming) para asegurar su viabilidad productiva.
"""

if third_dash != -1:
    content = content[:third_dash + 3] + contexto + content[third_dash + 3:]
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Avance 5 patched successfully.")
else:
    print("Could not find insertion point.")
