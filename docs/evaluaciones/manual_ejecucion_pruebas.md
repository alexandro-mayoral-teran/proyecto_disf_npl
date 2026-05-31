# Manual de Ejecución de Pruebas (Pipeline de Evaluación RAG)

Este manual te guiará paso a paso sobre cómo preparar, configurar y ejecutar las evaluaciones del sistema RAG, aprovechando nuestra arquitectura guiada por configuración (Config-Driven).

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
USE_LOCAL_LLM=true          # Activa el enrutador local/nube
USE_LOCAL_QA=true           # Privacidad en los documentos recuperados
USE_LOCAL_EXPANSION=true    # Ahorro en reescritura de queries
USE_LOCAL_EXTRACTION=false  # Extracción compleja JSON a cargo de GPT-4o
USE_LOCAL_JUDGE=false       # Evaluación rigurosa a cargo de GPT-4o
```

**Para pruebas 100% Nube (Máxima Precisión, Mayor Costo):**
Ideal para construir el "techo de cristal" en tu Frontera de Pareto.
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
2. `"exhaustivos"`: La matriz completa (ej. Baseline Léxico, Semántico, Híbrido, CrossEncoder, etc.) con la que construirás tu Frontera de Pareto.

**¿Cómo añadir un nuevo experimento?**
Simplemente agrega un nuevo bloque de diccionario al JSON. No necesitas tocar nada del código Python:
```json
{
    "nombre": "7_Estrategia_Nueva",
    "base_retriever": "hibrido",
    "query_expansion": "hyde",
    "post_processing": "cross_encoder"
}
```

---

## 3. Ejecución desde la Terminal

Una vez configurado el `.env` y tu `.json`, abre tu terminal, asegúrate de estar en la carpeta raíz del proyecto (`proyecto_disf_npl`) y ejecuta **uno** de los siguientes comandos:

### A) Modo Desarrollo (Rápido)
```bash
python src/lab/evaluador_integral.py --rapido
```
- **¿Qué hace?** Lee el bloque `"pruebas_rapidas"` del JSON. Evalúa únicamente **1 consulta** por cada fase (ideal para ahorrar tiempo y tokens mientras debugeas código).
- **¿Dónde guarda los resultados?** `data/03_output/evaluaciones/pruebas_rapidas/`

### B) Modo Pareto (Exhaustivo)
```bash
python src/lab/evaluador_integral.py --exhaustivo
```
- **¿Qué hace?** Lee el bloque `"exhaustivos"` del JSON. Ejecuta las **110 consultas** completas de tu dataset de evaluación, iterando por cada una de las estrategias que hayas definido.
- **¿Dónde guarda los resultados?** `data/03_output/evaluaciones/oficiales/`

### C) Modo Segmentado por Fases (Avanzado)
La Fase 3 (Análisis Desagregado) es muy pesada computacionalmente. Puedes prender y apagar fases a voluntad agregando flags (aplica tanto para `--rapido` como para `--exhaustivo`):

```bash
# Correr TODO (Mismo que no poner flags)
python src/lab/evaluador_integral.py --exhaustivo

# Correr SOLO Fase 1 (Arena) y Fase 2 (Contaminación Ciega)
python src/lab/evaluador_integral.py --exhaustivo --fase1 --fase2

# Correr SOLO Fase 3 (Análisis de Errores Desagregado)
python src/lab/evaluador_integral.py --exhaustivo --fase3
```

---

## 4. Revisión de Salidas

Una vez que el script finaliza (puede tardar desde un par de minutos hasta más de una hora en modo exhaustivo, dependiendo de si usas la Nube o Local), dirígete a la carpeta de salida correspondiente.

Deberás encontrar 3 archivos CSV inmutables (con *timestamp* de ejecución):
1. `ARENA_RESULTADOS_LLM_JUDGE_[fecha].csv` (Contiene Recall@10 y NDCG@10)
2. `contaminacion_ciega_[fecha].csv` (Validación de memoria del modelo)
3. `analisis_errores_desagregados_[fecha].csv` (Tipificación de errores A, B y C)

> [!TIP]
> Para interpretar qué significan las métricas de estos 3 archivos CSV, revisa el documento hermano: `support/guia_interpretacion_resultados.md`.
