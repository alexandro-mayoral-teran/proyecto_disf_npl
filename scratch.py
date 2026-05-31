import os

filepath = 'docs/entregas/Avance_4_Reporte_Final_Evaluaciones.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Let's search by string that we know is there
old_tax = '> 💡 **Análisis de la Taxonomía de Errores (Ejecución Local)**'

new_tax = '> 💡 **Análisis de la Taxonomía de Errores (Comparativa Nube vs Local)**\n> \n> Al analizar los resultados de la Fase 3, nos encontramos con un fenómeno documentado empíricamente en la literatura de MLOps conocido como **"LLM-as-a-Judge Leniency Bias" (Sesgo de Benevolencia del Juez)**. \n> \n> Si comparamos las corridas en la Nube (donde `gpt-4o` fungió como extractor y juez) contra las corridas Locales (donde `llama3.1` fungió como extractor y juez), observamos lo siguiente:\n> \n> *   **Errores A (Retrieval):** Idénticos en ambos escenarios (ej. 47 fallos). Esto demuestra consistencia determinista; ChromaDB y los Embeddings recuperaron los mismos fragmentos en ambos entornos independientemente del LLM generador.\n> *   **Errores B (Generación/Alucinación):** En la Nube, `gpt-4o` detectó **43 Errores tipo B**. En cambio, el Juez Local (`llama3.1`) detectó **0 Errores B** y catalogó casi todo el resto como "Éxito" absoluto.\n> \n> **¿Por qué la Nube detectó fallos masivos que el modelo Local ignoró?**\n> Esto no significa que el modelo local sea perfecto. Al contrario, demuestra el peligro de usar LLMs pequeños (como Llama de 8B parámetros) como **Jueces Evaluadores** en tareas complejas de extracción JSON. \n> \n> El modelo local evaluó *sus propias respuestas incompletas* y, al carecer de rigor analítico profundo, simplemente validó su propio trabajo como correcto (Éxito). En contraste, `gpt-4o` en la nube operó de manera implacable, detectando sutilezas lógicas y omisiones de campos frente al *Ground Truth* y catalogando las fallas como Error B.\n> \n> **Conclusión Estratégica:** Este hallazgo justifica arquitectónicamente nuestra decisión de aislar la validación y evaluación mediante "Model Cascading". Se recomienda encarecidamente utilizar Modelos Frontera (`gpt-4o`) para la telemetría evaluativa y como Jueces, dejando a los modelos locales enfocados exclusivamente en tareas operativas de baja fricción donde el riesgo sea tolerable.'

# find the start index of old_tax
start_idx = content.find(old_tax)
if start_idx != -1:
    # find the next section which is '---' after this block
    end_idx = content.find('---', start_idx)
    if end_idx != -1:
        content = content[:start_idx] + new_tax + '\n\n' + content[end_idx:]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print('Patched taxonomy successfully.')
else:
    print('Could not find taxonomy header.')
