# Manual Definitivo de Ejecución de Pruebas (Pipeline RAG - Avance 5)

Este manual te guiará paso a paso sobre cómo preparar, configurar y ejecutar las evaluaciones del sistema RAG, aprovechando nuestra arquitectura guiada por configuración (Config-Driven) y las nuevas capacidades.

---

## 1. Preparación del Entorno (`.env`)

Antes de correr cualquier evaluación, debes decidir **qué modelos** ejecutarán el pipeline. Todo se controla desde la raíz de tu proyecto en el archivo `.env`.

**Para pruebas 100% locales (Máxima Privacidad, Cero Costo):**
```env
USE_LOCAL_LLM=true
USE_LOCAL_QA=true
USE_LOCAL_EXPANSION=true
USE_LOCAL_EXTRACTION=true
USE_LOCAL_JUDGE=true
```
*(Asegúrate de que Ollama esté corriendo en tu computadora antes de ejecutar esto).*

**Para pruebas Híbridas (Recomendado para Producción):**
```env
USE_LOCAL_LLM=true          # Activa el enrutador local/nube (Cascade)
USE_LOCAL_QA=true           # Privacidad en los documentos recuperados
USE_LOCAL_EXPANSION=true    # Ahorro en reescritura de queries
USE_LOCAL_EXTRACTION=false  # Extracción compleja JSON a cargo de GPT-4o
USE_LOCAL_JUDGE=false       # Evaluación rigurosa a cargo de GPT-4o
```

**Para pruebas 100% Nube (Máxima Precisión, Mayor Costo):**
Ideal para construir el "techo de cristal" en la Frontera de Pareto.
```env
USE_LOCAL_LLM=false
USE_LOCAL_QA=false
USE_LOCAL_EXPANSION=false
USE_LOCAL_EXTRACTION=false
USE_LOCAL_JUDGE=false
```

---

## 2. Configuración de Experimentos (JSON)

Toda la matriz de pruebas (es decir, qué estrategias de búsqueda se van a comparar) vive exclusivamente en el archivo:
**`data/config_experimentos.json`**

Este archivo contiene dos bloques principales:
1. `"pruebas_rapidas"`: Ideal para validar que el código funciona después de un cambio. Solo debería tener 1 o 2 estrategias básicas.
2. `"exhaustivos"`: La matriz completa (ej. Baseline Léxico, Semántico, Híbrido, CrossEncoder, etc.) con la que se construirá la Frontera de Pareto.

**¿Cómo añadir un nuevo experimento?**
Simplemente agrega un nuevo bloque de diccionario al JSON. No se necesita tocar nada del código Python:
```json
{
    "nombre": "7_Estrategia_Nueva",
    "base_retriever": "hibrido",
    "query_expansion": "hyde",
    "post_processing": "cross_encoder"
}
```

---

## 3. Ejecución del Pipeline Integral (Evaluador)

Una vez configurado el `.env` y el `.json`, abre la terminal en la raíz del proyecto y ejecuta el evaluador integral.

> [!TIP]
> **Argumentos Disponibles (Flags):**
> - `--rapido`: Lee el bloque `"pruebas_rapidas"` del JSON (ideal para debugear).
> - `--exhaustivo`: Lee el bloque `"exhaustivos"` del JSON (usa todo tu dataset).
> - `--limite N`: Limita la prueba a las primeras `N` preguntas (ej. `--limite 5`).
> - `--tarea_eval [formato|qa]`: Determina qué tarea evaluar. `formato` evalúa extracción JSON; `qa` evalúa texto libre natural. (Default: `formato`).
> - `--fase1`, `--fase2`, `--fase3`: Te permiten correr módulos específicos. Si no pones ninguna, se corren todas.

### Ejemplos de uso comunes:

**A) Prueba Rápida de QA en Texto Libre (5 preguntas)**
```bash
python src/lab/evaluador_integral.py --rapido --limite 5 --tarea_eval qa
```

**B) Evaluación Completa para Frontera de Pareto**
```bash
python src/lab/evaluador_integral.py --exhaustivo
```

**C) Correr SOLO Análisis de Errores (Fase 3)**
```bash
python src/lab/evaluador_integral.py --exhaustivo --fase3
```

---

## 4. Scripts Satélite y Módulos Específicos

Se desarrollaron múltiples scripts independientes para auditorías avanzadas y validación estadística. Todos se ejecutan desde la terminal.

### 4.1. Probar el nuevo Router / Cascade por Confianza
Prueba interactivamente el Ensamble Heterogéneo (Llama 3.1 -> GPT-4o-mini). Si el modelo local tiene una fidelidad baja (< 0.8), el sistema escalará automáticamente la respuesta a la nube.
```python
import os
from src.nlp_core.generacion import responder_rag_cascade_qa

# Esto forzará el uso local primero, evaluará Faithfulness y luego decidirá
pregunta = "¿Cuáles son las sanciones por enviar tarde el reporte DISF?"
respuesta, telemetria, chunks = responder_rag_cascade_qa(pregunta, k=4)
print(telemetria)
```

### 4.2. Prueba de Calibración / Consistencia
Esta prueba interroga al modelo múltiples veces sobre la misma consulta (con Temperatura=0.7) para medir la varianza en sus respuestas factuales (Self-Consistency) y detectar miscalibración.
```bash
python src/lab/consistencia_eval.py
```

### 4.3. Comprobar Lift Estadístico (Bootstrap)
Evalúa estadísticamente (con un Intervalo de Confianza del 95%) si el salto cualitativo entre el Ensamble (Cascade) y el modelo de Baseline es matemáticamente significativo en un entorno real.
```bash
python src/lab/calcular_delta_ma4.py
```

### 4.4. Generar Frontera de Pareto Final
Traza la gráfica de Costo Operativo vs Calidad (NDCG@10) incluyendo todos los modelos evaluados en modo exhaustivo para demostrar empíricamente el valor financiero de la arquitectura.
```bash
python src/lab/generar_pareto_final.py
```

### 4.5. Pruebas de Penetración y Guardrails (Seguridad)
Evalúa el cortafuegos del sistema (`src/nlp_core/seguridad/guardrails.py`) enviándole un dataset malicioso (`eval_dataset_red_teaming.json`) para verificar si el Agente detecta y bloquea *Prompt Injections*, peticiones tóxicas, y *Jailbreaks* corporativos.
```bash
python src/lab/seguridad_eval.py
```

---

## 5. Auditoría y Taxonomía de Errores

Para aislar y auditar de dónde provienen las alucinaciones (Retrieval vs Generación), utilizamos el flujo:

### A) Generar Taxonomía Extendida (Juez LLM)
Usa un modelo Juez potente (GPT-4o) para inspeccionar por qué fallaron las respuestas en el evaluador integral y clasificarlas en Errores de Recuperación (A), Generación (B) o Formato (C).
```bash
python src/lab/extender_taxonomia_ma6.py
```
*(Se generará `taxonomia_extendida_MA6.csv` en la carpeta oficial).*

### B) Exportar Plantilla de Auditoría Humana
Filtra únicamente los Errores de Generación (Tipo B) detectados en el paso anterior y los exporta a Excel, listos para que un humano los clasifique (Útil, Parcial, Alucinación).
```bash
python src/lab/exportar_auditoria_manual.py
```
*(Se generará `auditoria_manual_generacion.xlsx` listo para usarse).*

---

## 6. Revisión de Salidas

Todos los resultados, logs y gráficos generados se guardan organizadamente en:
`data/03_output/evaluaciones/`

En la carpeta correspondiente a tu ejecución (`oficiales/run_...` o `pruebas_rapidas/run_...`) encontrarás archivos clave:
1. `ARENA_RESULTADOS_LLM_JUDGE_[fecha].csv`: Métricas de Recall, NDCG, Costo Total y Latencias P50/P95/P99.
2. `contaminacion_ciega_[fecha].csv`: Validación de memoria del LLM (Invariabilidad).
3. `taxonomia_extendida_MA6.csv`: Desglose automático de errores.
4. `auditoria_manual_generacion.xlsx`: Tu plantilla para validación humana.
5. `/graficos/pareto_avance5_cascade.png`: Tu visualización de Frontera de Pareto.
