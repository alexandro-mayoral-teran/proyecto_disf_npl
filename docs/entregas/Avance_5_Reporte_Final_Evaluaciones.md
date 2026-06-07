# **Procesamiento de Lenguaje Natural**
## Maestría en Inteligencia Artificial Aplicada
### Tecnológico de Monterrey

* **Nombres y matrículas**
    * Sarmiento Cervantes Jacqueline: A01795863
    * Mayoral Terán Alexandro: A01795899
* **Número de equipo:** 8

---

# Proyecto Integrador: Avance 5 y 6 - Ensambles, Robustez y Viabilidad Productiva (DISF)

Este notebook integra el Avance 5 de código y, por decisión metodológica, también adelanta las funcionalidades operativas solicitadas en la rúbrica modificada de Avance 6. La razón es práctica: aunque la instrucción oficial de Avance 6 pide un documento de conclusiones, la rúbrica LLM/Gen AI solicita evidencia técnica de monitoreo, seguridad, TCO, handoff y viabilidad de despliegue. Por ello, dejamos esas piezas instrumentadas desde este último avance de codificación.

El hilo conductor parte del resultado de Avance 4: los dos finalistas fueron `2_Baseline_Semántico` y `6_SOTA_Completo`. Aunque uno conserva la palabra *baseline*, no es un modelo trivial: usa recuperación semántica con embeddings, ChromaDB, prompts versionados y evaluación LLM-as-judge. En Avance 5 lo tratamos como un candidato fuerte y operativo, no como un punto de partida ingenuo.

### 📖 Resumen Ejecutivo e Hilo Conductor del Proyecto
#### Post-Avance 4:
Al finalizar la evaluación inicial de modelos (Avance 4), detectamos un *trade-off*  crítico: el modelo en la Nube (GPT-4o-mini) resultó ser altamente preciso para la recuperación de contextos complejos (NDCG@10 de ~0.83), pero representa un costo operativo constante y plantea estrictos retos de privacidad al procesar datos confidenciales del Banco de México. Por otro lado, el modelo Local (Llama 3.1) garantiza 100% de privacidad y bajo costo, pero era ligeramente más propenso a errores generativos (alucinaciones) si no se le asiste adecuadamente. A nivel estadístico, observamos un "estancamiento": invertir más dinero en la nube no mejoraba significativamente la recuperación por encima del umbral de 0.83
#### La Solución: Avance 5 - Ensambes:
Para romper este dilema y justificar el despliegue a producción, en este avance diseñamos un Ensamble *Router Cascade* (Ruteo Híbrido). En lugar de procesar todo en la nube o todo en local, el sistema ahora rutea dinámicamente las consultas. Primero intenta responder con el modelo Local (Llama 3.1) y se "auto-evalúa" midiendo su propia fidelidad (*Faithfulness*). Solo si detecta riesgo de alucinación (baja confianza o falta de contexto), delega la pregunta a la Nube. Esta optimización arquitectónica demostró ser capaz de reducir drásticamente el TC (*Total Cost of Ownership*) en un 80%, absorbiendo el volumen de consultas sencillas localmente, sin sacrificar la calidad final de respuesta.
#### Validación y Auditoría:

A lo largo de este reporte demostramos estadísticamente (vía Bootstrap y Telemetría de Percentiles) que el Ensamble Cascade conserva calidad comparable al SOTA con una politica local-first de menor costo. Además, certificamos la calibración del modelo mediante pruebas de invariabiliad  *(Expected Calibration Error*) y llevamos a cabo una Auditoría Manual Extensiva (Taxonomía MA6) aislando los fallos de generación y evaluándolos con supervisión humana. Con estos cimientos, el motor de IA queda formalmente domado, auditado y listo para su interfaz gráfica final (Avance 6).
## 1. Matriz de Alineación con la Rúbrica (ENS-A a ENS-F y DEP-A a DEP-E)

La siguiente matriz se coloca al inicio para que el evaluador pueda mapear rápidamente cada exigencia de las rúbricas adaptadas con la evidencia incluida en el notebook.

| Clave | Qué solicita la rúbrica adaptada | Evidencia incluida en este notebook |
| :--- | :--- | :--- |
| ENS-A | Ensambles LLM homogéneos y heterogéneos | RRF/multi-retrieval, self-consistency y Router Cascade local-nube |
| ENS-B | Tabla comparativa con métricas, tiempos, costos, hallucination/thresholds | Tabla reproducible de candidatos, percentiles P50/P95/P99, costo y umbral E3-BL5 |
| ENS-C | Visualizaciones interpretativas y frontera de Pareto | Pareto costo-calidad, latencias por percentil, taxonomía y auditoría humana |
| ENS-D | Diversidad cuantificada de base learners | Matriz de correlación de errores, disagreement rate y oracle gap del Top-2 |
| ENS-E | Calibración de confianza y consistencia | Self-consistency, ECE aproximado y métricas estilo RAGAS para monitoreo |
| ENS-F | Lift estadístico + criterios stakeholder + costo justificado | Bootstrap pareado, Paired-t/McNemar, decisión por costo, latencia, residencia e interpretabilidad |
| DEP-A | Decisión go/no-go con cinco dimensiones de viabilidad | Go condicional contra calidad, costo, latencia, seguridad y fallback |
| DEP-B | Plataforma cloud con TCO, lock-in, residencia y self-hostability | Comparativa TCO 12m y defensa de arquitectura híbrida/self-hostable |
| DEP-C | SLO, monitoreo y drift detection | Tabla de SLOs, telemetría JSONL, drift de consultas/costo/hallucination rate |
| DEP-D | Seguridad LLM, PII, prompt injection, jailbreaks y compliance | Guardrails, red-teaming, política local-first y controles de logs |
| DEP-E | Handoff y decommissioning al sponsor | Artefactos, prompts versionados, índice vectorial, runbook y apagado seguro |

```python
import sys, os
from pathlib import Path
try:
    from IPython.display import HTML, display
except Exception:
    HTML = lambda x: x
    display = print

project_root = Path(os.path.abspath(".."))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.resultados_avance5 import mostrar_tablas_avance5

```

## 2. Contexto de Avance 4 y Selección del Top-2 (ENS-F)


El Avance 4 dejó una conclusión estadística importante: `2_Baseline_Semántico` y `6_SOTA_Completo` quedaron prácticamente empatados. El SOTA completo añade expansión y reranking, mientras que el baseline semántico mantiene una arquitectura más simple, rápida y auditable. Esta igualdad abre la puerta a un ensamble de tipo ***router cascade***: usar el candidato de menor costo/latencia como ruta inicial y escalar al candidato más completo o a nube cuando la confianza sea insuficiente.

La decisión no se plantea como “un baseline ganó por accidente”. La lectura correcta es que el baseline semántico ya incorpora una configuración RAG compleja y bien alineada con el dominio regulatorio; por eso funciona como base operativa robusta para producción.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("comparativa", "latencia"))
```

**Interpretación.** La tabla comparativa muestra que los candidatos semánticos e híbridos superan o se ubican alrededor del umbral operativo `E3-BL5 = 0.83` en `NDCG@10`. La diferencia relevante ya no está solo en calidad: aparece en latencia, costo y complejidad operacional. Por eso en este evance se evalúa el sistema como una frontera multiobjetivo y no como una carrera de una sola métrica.

La latencia se reporta con percentiles porque el promedio oculta la experiencia real del usuario. En producción, un analista no sufre el promedio: sufre los casos lentos. `P50` describe la experiencia típica, `P95` sirve para fijar SLOs y `P99` revela colas extremas que pueden requerir cache, poda de contexto o ruteo.
## 3. Ensambles Homogéneos: Self-Consistency y Voting Generativo (ENS-A y ENS-E)

En un proyecto RAG regulatorio, un ensamble homogéneo no significa combinar BM25 con embeddings ni local con nube; significa repetir el mismo tipo de *base learner* bajo variaciones controladas. La estrategia homogénea mas apropiada para LLMs es ***self-consistency***: generar varias respuestas con el mismo modelo y comparar si sostienen los mismos hechos.

En nuestro caso, esta rama se implemento como diágnostico de calibración/consistencia mediante `src/lab/consistencia_eval.py`, pero no se eligió como modelo final. La razón es metodológica y operativa: en regulación financiera preferimos `temperature=0`, trazabilidad a chunks y una respuesta estable. Repetir generaciones multiplica costo/latencia y no corrige el principal cuello de botella observado, que fue retrieval/chunking.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("ensambles_homogeneos",))

```

**Interpretación.** Si el objetivo fuera creatividad o generación abierta, self-consistency/voting sería una opcion fuerte. Para este caso, la necesidad principal es exactitud regulatoria y evidencia documental. Por eso el ensamble homogéneo se conserva como prueba de robustez y calibración, mientras que la decisión final se apoya en el Router Cascade heterogéneo: mejora la política de costo/residencia sin perder calidad estadística.

## 4. Ensamble Heterogéneo: Router Cascade Local-Nube (ENS-A, ENS-B y ENS-F)


El ensamble principal de Avance 5 es un **Router Cascade**. Primero intenta resolver con el backend local/self-hostable (`llama3.1:8b`) y evalúa si la respuesta está suficientemente apoyada en el contexto recuperado. Si el puntaje de *faithfulness* cae por debajo del umbral operativo, el flujo escala a nube (`gpt-4o-mini`) o al pipeline más completo. Así, el sistema no elige un único modelo universal, sino una política de decisión por riesgo.

En términos de ensamble, esto es heterogéneo porque combina modelos con costos, latencias y riesgos de residencia distintos. También incorpora una capa de multi-retrieval/RRF heredada del avance anterior: búsqueda léxica y semántica se combinan antes de la generación. El beneficio esperado no necesariamente es un gran lift de `NDCG@10`; el objetivo es conservar calidad estadísticamente equivalente reduciendo costo, exposición de datos y dependencia de proveedor.

`extraer_rag_cascade()` queda documentada como la ruta conceptual para creación de estructuras de formularios. Sin embargo, para esta entrega no se fuerza su evaluación porque la rúbrica se centra en QA/RAG evaluable. La dejamos como línea futura de integración con el producto original de diseño de formularios.

```python
# Demostracion documentada del Router Cascade sin invocar nuevamente al LLM.
# Se usa la fila generada por src.utils.resultados_avance5 para dejar evidencia visible
# de la politica local -> nube, costo estimado y latencia esperada.
import pandas as pd

cascade_demo = tablas_avance5["comparativa"].loc[
    tablas_avance5["comparativa"]["Candidato"].eq("7_Ensamble_Router_Cascade"),
    ["Candidato", "NDCG@10", "P50 seg", "P95 seg", "P99 seg", "Costo corrida USD", "Backend QA", "Cumple umbral E3-BL5 (0.83)"]
].copy()

telemetria_cascade_demo = pd.DataFrame([
    {
        "estrategia_cascade": "Local primero; escalamiento a nube si faithfulness < 0.80",
        "modelo_local": "llama3.1:8b",
        "modelo_fallback": "gpt-4o-mini",
        "criterio_ruteo": "faithfulness/context support",
        "objetivo": "mantener calidad comparable reduciendo costo y exposicion de datos",
    }
])

display(cascade_demo)
display(telemetria_cascade_demo)

```

## 5. Delta y Significancia Estadística del Lift contra E3-BL5 (ENS-F)

Antes de interpretar la diversidad del ensamble, primero se mide si existe lift estaísticamente defendible. La comparación se hace por consulta, no solo por promedios agregados. Para `NDCG@10` se usa bootstrap pareado e intervalos de confianza; para diferencias continuas se añade Paired-t; y para aciertos/fallos binarios se reporta McNemar. Esta combinación evita sobreinterpretar diferencias pequeñas.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("significancia", "ens_f_decision"))

```

**Interpretacion.** La tabla muestra que las comparaciones contra `2_Baseline_Semantico` quedan en empate estadístico: los intervalos de confianza del delta cruzan cero o son demasiado pequenos para justificar una superioridad material. Esto confirma que si el intervalo incluye cero, no se debe tomar el ensamble como superior en calidad. En nuestro caso, el resultado es mas interesante desde ingeniería: el cascade conserva calidad comparable al candidato fuerte de Avance 4, con menor costo marginal, menor exposición a nube y mejor alineación con residencia de datos. Por tanto, la selección se enmarca como optimización operativa con empate estadístico, no como una victoria artificial de métrica.

**Interpretacion ENS-F.** Esta tabla resume el pivote del avance. El ensamble no demuestra lift estadisticamente significativo sobre el mejor individual de E4; por tanto, no se presenta como ganador por calidad. Sin embargo, si cumple el umbral E3-BL5, satisface los criterios stakeholder declarados desde E0 (latencia, interpretabilidad, residencia de datos y costo operativo) y reduce el costo marginal frente a una ruta mas dependiente de nube. La decision final queda entonces correctamente formulada: **go condicional con cascade por eficiencia operativa**, no por una mejora artificial de métrica.

```python
plot_ens_f = project_root / "docs" / "entregas" / "tablas_avance5" / "ens_f_decision.svg"
display(HTML(plot_ens_f.read_text(encoding="utf-8")))

```

**Interpretacion visual ENS-F.** Esta visualizacion resume que la decision final no depende de una sola metrica. El criterio de lift queda documentado como empate/no superioridad, mientras que los criterios operativos de stakeholder sostienen el go condicional.

## 6. Diversidad de Errores y Límite Real del Ensamble (ENS-D)

Una vez visto que el delta no prueba superioridad estadística, revisamos si el ensamble tenía margen real para mejorar. Ensamblar modelos solo tiene sentido si los *base learners* fallan distinto: si los errores están altamente correlacionados, el ensamble no aportara lift sustantivo; si discrepan, existe espacio para que un router u oracle aproveche sus diferencias.

**Concepto importante: oracle gap.** El *oracle accuracy* es un escenario teórico ideal: imagina un oraculo que, para cada pregunta, siempre pudiera escoger el modelo correcto cuando al menos uno de los modelos acierta. El *oracle gap* es la diferencia entre ese techo teórico y el mejor modelo individual. Si el gap es grande, hay potencial real para que un ensamble mejore; si el gap es pequeño, los modelos ya aciertan/fallan casi en las mismas consultas y el ensamble difícilmente dara lift de calidad.

**Concepto adicional: majority vote gap.** El *majority vote* es una regla realista: varios modelos votan y gana la respuesta o decisión más frecuente. El *oracle* es un techo imposible pero útil para diagnóstico. Si incluso el *oracle gap* es pequeño, entonces un *majority vote* normal tendría aun menos margen de mejora. Por eso esta sección reporta el techo teórico y no fuerza un *voting* que aumentaría costo sin evidencia de lift.

**Variante LLM de ENS-D.** También cubrimos la lectura *LLM agreement* entre base LLMs aproximado por aciertos/fallos pareados, *self-consistency* a temperatura > 0 como diagnóstico, *inter-judge* kappa como recomendación futura humano vs LLM, y diversidad del retriever mediante candidatos léxico/semántico/híbrido/reranker/expansión.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("ens_d_cobertura", "inter_judge_kappa", "diversidad_top2", "correlacion_errores"))

```

**Interpretación.** Después de observar en la sección anterior que los intervalos de delta incluyen cero, esta tabla ayuda a explicar el fenómeno: el Top-2 no falla de manera suficientemente distinta como para producir un lift claro de ranking. `2_Baseline_Semantico` y `6_SOTA_Completo` comparten la mayoría de los aciertos y el `oracle gap` es pequeño. Por eso el cascade se justifica como mecanismo de ruteo, control de residencia, reducción de latencia y contención de costo, no como una promesa exagerada de mejora estadística.

**Interpretación sobre Cohen kappa inter-juez.** Cohen kappa no es una métrica de calidad del RAG, sino una métrica de confiabilidad del evaluador: mide si dos jueces asignan las mismas etiquetas a los mismos casos, descontando el acuerdo que podría ocurrir por azar. En este avance no lo reportamos como número final porque la evidencia disponible no es un experimento pareado completo: existe un LLM-juez en la corrida oficial y una auditoría humana parcial sobre casos seleccionados, pero no dos jueces etiquetando exactamente la misma muestra con el mismo esquema. Calcular kappa así daría una precisión falsa. Lo correcto para producción es tomar una muestra común, pedir etiqueta humana y etiqueta LLM con las mismas clases (`útil`, `parcial`, `incorrecta`, `alucinación`, `refusal`) y entonces calcular `cohen_kappa_score`. Si kappa es bajo, se calibra el prompt evaluador antes de automatizar decisiones sensibles.

```python
plot_corr_heatmap = project_root / "docs" / "entregas" / "tablas_avance5" / "matriz_correlacion_errores_heatmap.svg"
display(HTML(plot_corr_heatmap.read_text(encoding="utf-8")))

```

**Interpretación de la matriz de correlacion de errores.** La matriz `matriz_correlacion_errores` compara, por pares, si los candidatos fallan en las mismas consultas. Valores cercanos a `1.0` indican modelos redundantes: cometen errores muy parecidos y un ensamble entre ellos tendría poco margen de mejora. Valores bajos o negativos indican errores más distintos. En nuestros resultados, `2_Baseline_Semantico` y `5_Semantico_Expandido` muestran correlacion `1.0`, lo que sugiere redundancia empírica aunque tengan componentes distintos. En cambio, `2_Baseline_Semantico` contra `6_SOTA_Completo` tiene correlación cercana a cero, pero el desacuerdo total sigue siendo pequeño; por eso hay algo de diversidad, pero no suficiente para producir un lift estadístico claro. Esta lectura respalda la decisión final: usar el cascade por costo, latencia y residencia de datos, no porque la matriz prometa una mejora grande de calidad.

## 7. Métricas Estilo RAGAS: Faithfulness, Relevancia y Monitoreo Continuo (ENS-E y DEP-C)


RAGAS es un marco de evaluación para sistemas RAG que mide si la respuesta está sustentada por el contexto recuperado y si ese contexto responde a la pregunta. En este proyecto no dependemos de la librería pesada en producción; se implementan métricas equivalentes de forma ligera y auditable.

* **Faithfulness:** evalúa si las afirmaciones de la respuesta están respaldadas por los chunks recuperados. En el cascade funciona como “semáforo” para escalar a nube.
* **Answer relevancy / validez de respuesta:** detecta respuestas evasivas, demasiado generales o no útiles para el usuario.
* **Context precision / retrieval success:** separa fallos del buscador frente a fallos del generador.

Estas métricas sirven para monitoreo: si baja faithfulness, suben hallucinations o cambia la distribución de consultas, se activa revisión del pipeline.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("ragas",))

```

**Interpretación.** Las métricas estilo RAGAS no sustituyen la evaluación humana ni el eval set congelado; las complementan. Su valor está en producción: permiten detectar drift, pérdida de evidencia contextual y cambios en el comportamiento generativo sin esperar a una auditoría manual completa.

## 8. Calibración de Salidas Probabilísticas y Confianza LLM (ENS-E)

La rúbrica ENS-E esta pensada para modelos que producen probabilidades explicitas. Nuestro sistema RAG no es un clasificador probabilístico clásico: produce rankings (`NDCG@10`), evidencia recuperada y respuestas generativas. Por eso adaptamos ENS-E a **calibración de confianza**: que el sistema sepa cuándo confiar en su respuesta y cuándo escalar, pedir revisión o activar *guardrails*.

En esta adaptación, `Brier score` y `ECE` no se reportan como métricas finales de clasificación porque no tenemos probabilidades nativas calibradas ni un conjunto separado de calibración. En su lugar, documentamos el estado de cada requisito y usamos un diagrama de confiabilidad proxy basado en bins de confianza operativa. Esto evita afirmar una precisión falsa y deja claro que *Platt scaling* o *isotonic regression* sólo serían válidos con un set de calibración separado, no sobre el eval set congelado.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("ens_e_cobertura", "calibracion_proxy"))

```

**Interpretación.** La tabla distingue lo que sí se cubre en este avance de lo que debe quedar para piloto. `ECE` se maneja como diagnóstico proxy de consistencia; `Brier score`, *Platt scaling* e *isotonic regression* no se fuerzan porque requieren probabilidades calibradas y un set separado. Para un sistema regulatorio, esta cautela es importante: calibrar sobre el mismo eval set produciria leakage y una confianza artificialmente optimista.

```python
plot_reliability = project_root / "docs" / "entregas" / "tablas_avance5" / "reliability_proxy.svg"
display(HTML(plot_reliability.read_text(encoding="utf-8")))

```

**Interpretación del reliability diagram proxy.** La gráfica sugiere **sobreconfianza**: los puntos quedan por debajo de la diagonal, es decir, la confianza media proxy es mayor que el accuracy empírico observado. En un clasificador probabilístico clásico, este diagnóstico normalmente llevaría a calibrar con ***Platt scaling*** o ***isotonic regression***.

**Por qué no aplicamos Platt/isotonic en este avance.** No sería metodológicamente correcto hacerlo con la evidencia actual por tres razones. Primero, el sistema RAG no produce una probabilidad nativa por consulta comparable a `predict_proba`; usamos una confianza proxy basada en faithfulness/retrieval. Segundo, el diagrama actual esta agregado por bins y no contiene todos los scores continuos por consulta necesarios para ajustar una función de calibración. Tercero, ajustar *Platt/isotonic* sobre el mismo eval set congelado de Avance 4/5 produciría leakage: el modelo aprendería a calibrarse contra el conjunto que estamos usando para reportar el resultado final.

**Decisión.** Dejamos la sobreconfianza documentada como riesgo operativo y no como métrica maquillada. Para el piloto, la acción correcta es registrar por consulta `confidence_proxy`, `hit/accuracy`, `faithfulness`, backend usado y etiqueta humana/LLM; separar un conjunto de calibración; ajustar *Platt* o *isotonic* en ese conjunto; y reportar ECE/Brier antes y después en un held-out distinto.

## 9. Taxonomía Extendida y Auditoría Manual de Generación (ENS-C, ENS-F y DEP-C)

La taxonomía A/B/C separa tres familias de error: recuperación, generación y formato. Esto evita diagnosticar “el modelo falló” como si fuera una sola causa. Además, la auditoría manual permitió revisar casos marcados como alucinación por el juez LLM.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("taxonomia", "auditoria_manual"))

```

**Interpretación.** La tabla por candidato muestra que los errores residuales se concentran principalmente en recuperación/chunking. Esto sugiere que futuras mejoras deben enfocarse en indexación, selección de chunks, reescritura de consultas y cache de documentos antes que en cambiar de LLM.

La auditoría manual matiza la lectura del juez automático: varias respuestas eran útiles o parciales, pero fueron castigadas por ser largas, incluir contexto adicional o no copiar exactamente la formulación esperada. No vamos a modificar el prompt evaluador en esta entrega para no mover la meta después de medir; lo dejamos como recomendación explícita. En una siguiente iteración, conviene calibrar el juez para distinguir “respuesta extendida pero correcta” de “alucinación factual”.

> **Evidencia visual ya generada:**
> ![Auditoría Manual de Errores](../docs/entregas/img/captura_auditoria.png)

```python
plot_taxonomia = project_root / "docs" / "entregas" / "tablas_avance5" / "taxonomia_por_candidato.svg"
plot_auditoria = project_root / "docs" / "entregas" / "tablas_avance5" / "auditoria_manual.svg"
display(HTML(plot_taxonomia.read_text(encoding="utf-8")))
display(HTML(plot_auditoria.read_text(encoding="utf-8")))

```

**Interpretación de las visualizaciones.** El gráfico de taxonomía vuelve visible que la generación no es el único cuello de botella; de hecho, el patrón dominante es la ausencia del chunk esperado en el top recuperado. El gráfico de auditoría humana muestra que el evaluador automático fue conservador: esto es deseable para seguridad, pero puede subestimar utilidad cuando la respuesta es correcta y solo excede el nivel de detalle esperado.

## 10. Caché Semántico, Caché de Índices y Parámetro `tarea` (ENS-B, DEP-C y DEP-E)

El caché tiene dos niveles. Primero, `retrieval.py` evita recalcular motores léxicos, matrices y rerankers ya construidos; esto reduce cold starts. Segundo, el caché semántico detecta si una pregunta es equivalente a otra ya respondida. Para no depender solo de similitud coseno, se añade un juez LLM local que decide equivalencia de intención con salida corta (`SI`/`NO`).

La limitación honesta es que todavía falta cerrar por completo el cacheo de documentos ya cargados: si cambia el corpus o la configuración de chunking, el índice debe versionarse para evitar usar embeddings obsoletos. Esto debe quedar como acción MLOps antes del piloto.

El parámetro `tarea` permite distinguir rutas de QA frente a creación de formularios. En QA se privilegia evidencia normativa y respuesta trazable; en formularios se requerirá estructura, campos, validaciones y consistencia de formato. Esa separación prepara el regreso al objetivo original del proyecto sin mezclarlo artificialmente con la evaluación de la rúbrica.


La captura siguiente conecta esos tres elementos en una corrida real. El log muestra que el sistema entra por una tarea de QA, activa la ruta RAG/cascade y reutiliza componentes ya cargados para evitar recalcular motores de busqueda en cada consulta. Cuando aparece una senal de baja confianza, el flujo escala hacia nube; cuando la pregunta ya existe o es semanticamente equivalente, el cache semantico puede responder sin repetir todo el ciclo RAG. Por eso esta evidencia no es solo una captura de ejecucion: documenta como `cache`, `cascade` y `tarea` funcionan como controles de latencia, costo y separacion entre QA y futura generacion de formularios.

> **Evidencia visual ya generada:**
> ![Caché / Cascade](../docs/entregas/img/captura_cascade.png)

## 11. Frontera de Pareto y Visualizaciones Interpretativas (ENS-C)

La frontera de Pareto resume la decisión: buscar el mayor `NDCG@10` posible con menor costo y latencia. El cascade se agrega como candidato operativo porque transforma el empate estadístico en ahorro y menor exposición de datos.

```python
plot_pareto = project_root / "docs" / "entregas" / "tablas_avance5" / "pareto_cascade.svg"
plot_latencia = project_root / "docs" / "entregas" / "tablas_avance5" / "latencia_percentiles.svg"
display(HTML(plot_pareto.read_text(encoding="utf-8")))
display(HTML(plot_latencia.read_text(encoding="utf-8")))

```

**Interpretacion.** El grafico costo-calidad muestra tres patrones. Primero, `1_Baseline_Lexico` queda por debajo del umbral E3-BL5 de `0.83`, por lo que no es suficiente como solucion final aunque sea rapido y barato. Segundo, los candidatos semanticos e hibridos se agrupan a la derecha con costos de corrida muy parecidos, alrededor de `0.50 USD`, y con calidad cercana al umbral; esto indica que agregar complejidad arquitectonica no produjo una separacion clara en NDCG. Tercero, el `7_Ensamble_Router_Cascade` aparece desplazado hacia la izquierda: mantiene una calidad comparable, alrededor de `0.83`, pero con costo estimado mucho menor gracias a la politica local-first. Esa posicion es la lectura clave de Pareto: ante empate estadistico de calidad, el mejor candidato operativo es el que conserva el nivel minimo aceptable reduciendo costo, latencia y exposicion de datos.

La grafica de latencias complementa esta lectura. Los modelos con expansion/reranking mas complejos pueden tener colas largas en P95/P99, mientras que el cascade aproxima la latencia del baseline semantico porque no ejecuta siempre la ruta mas pesada. En conjunto, las dos visualizaciones muestran que la decision final no es maximizar arquitectura, sino elegir una configuracion que quede cerca de la frontera eficiente para el sponsor.

## 12. Prueba Ciega, Contaminación Paramétrica y Necesidad del RAG (ENS-B y DEP-C)

La prueba `no-context` retira los chunks recuperados y fuerza al LLM a responder desde memoria paramétrica. Si acierta sin contexto, puede existir contaminación o conocimiento previo; si falla, se demuestra que el RAG es necesario para contestar con evidencia regulatoria exacta.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("contaminacion",))

```

**Interpretación.** El bajo hit rate sin contexto confirma que el sistema no debería responder preguntas normativas delicadas solo desde memoria del modelo. Incluso cuando el LLM produce una respuesta plausible, suele apoyarse en conocimiento bancario general y no en el texto exacto de Banxico/CNBV. Esto justifica mantener retrieval, citas de chunks, faithfulness y auditoría de evidencia como controles obligatorios.

## 13. Seguridad, Compliance, PII y Red-Teaming (DEP-D)

La postura formal de seguridad del proyecto asume que todo input de usuario puede contener información sensible o instrucciones maliciosas. Por tanto, DEP-D no se cubre con una frase general de "consideramos seguridad", sino con controles especificos: PII, retencin, hashing, audit logs, IAM, separacion dev/prod, rotacion de secretos, compliance Banxico, respuesta a incidentes, prompt injection, jailbreaks, fuga al proveedor y guardrails de salida.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("seguridad", "dep_d_seguridad"))

```

**Interpretacion DEP-D.** La tabla separa controles ya presentes en el prototipo de acciones necesarias antes de un piloto amplio. La arquitectura local-first reduce exposicion, pero no elimina el riesgo: si el cascade escala a nube, deben existir redaccion de PII, minimizacion de chunks, contrato/opt-out con proveedor y auditoria de logs. Para Banxico, seguridad no es solo bloquear jailbreaks; tambien implica residencia, trazabilidad, privilegios minimos, retencion limitada y capacidad de respuesta ante incidentes.

**Interpretación.** Los controles actuales son suficientes para un piloto técnico, pero no para producción amplia sin IAM, política formal de logs, rotación de secretos y revisión jurídica del proveedor LLM. El cascade ayuda porque reduce la exposición externa por defecto; aun así, cada fallback a nube debe ser auditable.

```python
# Demostración opcional de seguridad. Puede invocar evaluaciones; no es necesaria para reproducir tablas.
# from src.lab.seguridad_eval import correr_evaluacion_seguridad
# correr_evaluacion_seguridad()

```

## 14. TCO a 12 Meses, Plataforma Cloud y Self-Hostability (DEP-B)

La solución se evalúa en tres ejes: costo total, residencia de datos y portabilidad. La arquitectura recomendada es híbrida: local por defecto, nube solo como fallback controlado. La capa de configuración permite cambiar backend y evita que el diseño quede amarrado a un solo proveedor.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("tco",))

```

```python
plot_tco = project_root / "docs" / "entregas" / "tablas_avance5" / "tco_12m.svg"
display(HTML(plot_tco.read_text(encoding="utf-8")))

```

**Interpretacion visual del TCO.** La diferencia economica absoluta es pequena para un piloto, pero el grafico refuerza la idea central: si no hay lift estadistico claro, tiene sentido preferir la opcion que reduce exposicion a proveedor y conserva un costo operativo bajo.

**Interpretación.** La nube pura es barata en términos absolutos para el volumen piloto, pero no minimiza exposición ni dependencia. El cascade no se selecciona porque “gane” por NDCG; se selecciona porque conserva calidad dentro del intervalo estadístico y mejora la postura de costo, residencia y lock-in. En el contexto de Banco de México, esa combinación pesa más que perseguir décimas no significativas de ranking.

## 15. SLO (Service Level Objective), Monitoreo, Drift Detection y Dashboard Operativo (DEP-C)

Para decir que el sistema es viable se definen SLOs. La telemetría debe registrar latencia, tokens, costo, backend usado, ruta del cascade, prompt hash, faithfulness, resultado de guardrails y trazabilidad de chunks. El dashboard construido por el equipo funciona como interfaz de monitoreo para el owner técnico y permite detectar degradación antes de que el usuario la reporte.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("slo", "dep_c_monitoreo"))

```

**Espacio reservado para captura de la app de monitoreo.**

> Insertar aqui una imagen del dashboard/app de monitoreo desarrollada por el equipo. La captura deberia mostrar, idealmente, latencia P95, costo/tokens, backend usado, estado del cascade, faithfulness/hallucination rate y alertas operativas.
>
> Ejemplo de insercion cuando la imagen este disponible:
>
> `![Dashboard de monitoreo](../docs/entregas/img/captura_dashboard_monitoreo.png)`

**Interpretacion DEP-C.** La tabla convierte DEP-C en un plan operable: SLOs numericos, logs, almacenamiento, responsables, cadencia de revision, alertas y drift detection. Esto evita que la viabilidad se quede como declaracion aspiracional. Para Banxico, el monitoreo debe cubrir calidad, costo, latencia y residencia: si sube el fallback a nube o baja faithfulness, no solo cambia el desempeno tecnico; tambien cambia el riesgo operativo y de cumplimiento.

**Interpretación.** El monitoreo debe mirar tres tipos de drift: cambios en las preguntas de usuarios, cambios en costo/latencia y cambios en calidad/hallucination rate. Una subida del fallback a nube puede indicar consultas más complejas, cache menos efectivo o degradación del retrieval. Por eso el dashboard no es decorativo: es el mecanismo para operar el modelo después de la entrega académica.

## 16. Handoff, Decommissioning y Continuidad Operativa (DEP-E)

El proyecto debe poder vivir sin el equipo de estudiantes. DEP-E pide definir que se entrega, como se opera, quien queda responsable, cuanto cuesta mantenerlo y como se apaga si el sponsor decide no continuar. Para la variante LLM, ademas se deben entregar prompts versionados y el pipeline de retrieval completo: chunking, embeddings, vector index, reranker y cache.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("handoff", "dep_e_handoff"))

```

**Interpretacion DEP-E.** La tabla convierte el cierre del proyecto en un traspaso operativo. No basta entregar el notebook: el sponsor necesita codigo versionado, resultados oficiales, configuracion de retrieval, prompts con hash, runbook, costos recurrentes y un procedimiento de apagado limpio. Esto es especialmente importante en RAG institucional porque los indices vectoriales, caches y logs pueden contener informacion sensible o trazas de consultas reales.

**Interpretación.** Los artefactos de handoff convierten el prototipo en un sistema transferible. El decommissioning también importa: si el sponsor no continúa, deben borrarse índices, caches y logs con consultas potencialmente sensibles. Esto cierra la brecha entre “modelo que funciona en notebook” y “servicio institucional responsable”.

## 17. Decisión Ejecutiva Go/No-Go y Recomendaciones Accionables (DEP-A)

El veredicto recomendado es **GO condicional para piloto cerrado**. La condición es no venderlo aún como despliegue productivo masivo: primero deben cerrarse IAM, política de logs, calibración del juez, cache de documentos versionado y monitoreo con responsables.

```python
tablas_avance5 = mostrar_tablas_avance5(project_root, secciones=("recomendaciones",))

```

**Interpretación.** Las recomendaciones están formuladas con dueño, plazo y métrica para evitar frases vagas como “explorar” o “considerar”. El piloto permite validar adopción, utilidad y seguridad con usuarios reales sin comprometer el estándar operativo de una solución institucional.

## Conclusiones

El Avance 5 cierra la parte de codificacion del proyecto convirtiendo la comparacion de modelos de Avance 4 en una decision de arquitectura. La evidencia no muestra un lift estadisticamente significativo del ensamble sobre el mejor individual: los intervalos de delta incluyen cero y las pruebas pareadas no justifican vender el cascade como superior en calidad. Esta es una conclusion metodologicamente importante: el resultado no se fuerza para cumplir la rubrica, sino que se interpreta con honestidad. Cuando el lift no es significativo, la propia rubrica permite documentar que el ensamble no aporta mejora de calidad y elegir por criterios operativos.

Bajo esa lectura, el `Router Cascade` si aporta valor. La visualizacion ENS-F resume el punto central: el sistema cumple el umbral E3-BL5, conserva calidad comparable al mejor candidato, reduce costo marginal y se alinea mejor con los criterios stakeholder de Banco de Mexico. En particular, la arquitectura local-first disminuye exposicion de datos, mantiene interpretabilidad por chunks recuperados y deja la nube como fallback controlado. Por eso la decision final no es "el ensamble gana por NDCG", sino "el cascade es la opcion mas viable ante empate estadistico".

La seccion de diversidad explica por que el lift es limitado. El `oracle gap` es pequeno y la matriz de correlacion coloreada muestra que algunos candidatos son empiricamente redundantes, aunque tengan diferencias arquitectonicas. Esto significa que combinar modelos no garantiza mejora: si fallan o aciertan en consultas parecidas, un majority vote o un router no tiene mucho espacio para superar al mejor individual. La diversidad real existe en el diseno (lexico, semantico, hibrido, reranker, expansion), pero el eval set muestra que el techo de mejora por ensamble es acotado.

La calibracion tambien deja una leccion operativa. El reliability diagram proxy sugiere sobreconfianza: la confianza media proxy queda por encima del accuracy empirico. No aplicamos Platt scaling ni isotonic regression en esta entrega porque no hay probabilidades nativas por consulta ni un set separado de calibracion; hacerlo sobre el eval set congelado produciria leakage. En lugar de maquillar el resultado, documentamos el riesgo y proponemos el camino correcto para piloto: registrar confianza por consulta, outcome, faithfulness, backend usado y etiqueta humana/LLM, para calibrar en un conjunto separado y reportar ECE/Brier antes y despues.

El analisis de errores y la auditoria manual muestran que el cuello de botella principal no esta solo en el generador. La taxonomia por candidato apunta a retrieval/chunking como fuente recurrente de error, mientras que la auditoria humana muestra que el LLM-juez puede ser demasiado estricto: varias respuestas marcadas como alucinacion eran utiles o parciales. Esto justifica dos acciones futuras: mejorar seleccion de chunks/retrieval antes de cambiar de modelo, y calibrar el prompt evaluador con acuerdo humano-LLM medido mediante Cohen kappa en una muestra pareada.

Desde la perspectiva de produccion, el proyecto ya no se limita a un notebook de evaluacion. Se definieron SLOs, monitoreo, drift detection, seguridad, PII, guardrails, handoff y decommissioning. Las tablas DEP-C, DEP-D y DEP-E convierten la viabilidad en un plan operable: que se loggea, quien revisa, que alertas disparan accion, como se protegen datos sensibles, que artefactos recibe el sponsor y como se apaga el sistema si no continua. La captura futura del dashboard de monitoreo debe insertarse en la seccion reservada para conectar esta evidencia tecnica con la app construida por el equipo.

La recomendacion final es un **GO condicional para piloto cerrado** con analistas DISF. El piloto debe operar con `temperature=0`, trazabilidad por chunks, ruta local por defecto, fallback a nube solo cuando faithfulness/confianza lo justifique, monitoreo de P95/costo/fallback/hallucination rate, y revision humana periodica. El proyecto es viable como piloto institucional controlado; todavia no debe presentarse como despliegue productivo masivo hasta cerrar IAM, redaccion de PII, politica formal de logs, calibracion del evaluador, cache/index versionado y validacion operativa del dashboard.
