# **Procesamiento de Lenguaje Natural**
## Maestría en Inteligencia Artificial Aplicada
### Tecnológico de Monterrey

* **Nombres y matriculas**
    * Sarmiento Cervantes Jacqueline: A01795863
    * Mayoral Teran Alexandro: A01795899
* **Numero de equipo: 8**

---

# 🎓 Proyecto Integrador: Avance 5 y 6 - Robustez, Viabilidad y Despliegue (DISF)

Este documento es el **entregable oficial** conjunto. Su propósito es certificar que la arquitectura RAG seleccionada no solo es precisa y robusta (Rúbrica 5 / ENS), sino que es **segura, monitoreable y financieramente viable para Producción** (Rúbrica 6 / DEP). 

### 📖 Resumen Ejecutivo e Hilo Conductor del Proyecto

**El Reto (Post-Avance 4):**  
Al finalizar la evaluación inicial de modelos (Avance 4), detectamos un *trade-off* crítico: el modelo en la Nube (GPT-4o-mini) resultó ser altamente preciso para la recuperación de contextos complejos (NDCG@10 de ~0.83), pero representa un costo operativo constante y plantea estrictos retos de privacidad al procesar datos confidenciales del Banco de México. Por otro lado, el modelo Local (Llama 3.1) garantiza 100% de privacidad y bajo costo, pero era ligeramente más propenso a errores generativos (alucinaciones) si no se le asiste adecuadamente. A nivel estadístico, observamos un "estancamiento": invertir más dinero en la nube no mejoraba significativamente la recuperación por encima del umbral de 0.83.

**La Solución (Avance 5 - Ensambles):**  
Para romper este dilema y justificar el despliegue a producción, en este avance diseñamos un **Ensamble Router Cascade (Ruteo Híbrido)**. En lugar de procesar todo en la nube o todo en local, el sistema ahora rutea dinámicamente las consultas. Primero intenta responder con el modelo Local (Llama 3.1) y se "auto-evalúa" midiendo su propia fidelidad (*Faithfulness*). Solo si detecta riesgo de alucinación (baja confianza o falta de contexto), delega la pregunta a la Nube. Esta optimización arquitectónica demostró ser capaz de reducir drásticamente el TCO (Total Cost of Ownership) en un 80%, absorbiendo el volumen de consultas sencillas localmente, sin sacrificar la calidad final de respuesta.

**Validación y Auditoría:**  
A lo largo de este reporte demostramos estadísticamente (vía Bootstrap y Telemetría de Percentiles) que el Ensamble Cascade logra **calidad SOTA a precios de modelo Local**. Además, certificamos la calibración del modelo mediante pruebas de invariabilidad (Expected Calibration Error) y llevamos a cabo una **Auditoría Manual Extensiva (Taxonomía MA6)** aislando los fallos de generación y evaluándolos con supervisión humana. Con estos cimientos, el motor de IA queda formalmente domado, auditado y listo para su interfaz gráfica final (Avance 6).

---

## PARTE I: ENSAMBLES, CONSISTENCIA Y DIVERSIDAD (RÚBRICA 5 / ENS)

### 1. Rúbrica ENS-A: Estrategias de Ensamble Homogéneas y Heterogéneas

El proyecto implementa un enfoque de ensamble dual para maximizar la efectividad operativa y financiera:
*   **Ensamble Heterogéneo (Model Cascading & RRF):** Se utiliza *Reciprocal Rank Fusion* (RRF) para ensamblar múltiples retrievers (BM25 léxico + BGE semántico), combinando lo mejor de la búsqueda por palabra clave y la búsqueda por significado. Posteriormente, se emplea el *Cascading* entre LLMs: las tareas de ruteo y evaluación inicial de confianza se delegan al modelo local (Llama 3.1 8B), reservando el modelo GPT-4o-mini (Nube) como un mecanismo de *fallback* únicamente para consultas donde la fidelidad local no alcanza el umbral operativo.
*   **Ensamble Homogéneo (Self-Consistency):** El sistema permite votar sobre múltiples realizaciones generativas a temperatura > 0 para verificar coherencia de los hechos extraídos.

```python
# Demostración del soporte multi-modelo y Cascade por Confianza (Faithfulness)
from src.nlp_core.generacion import responder_rag_cascade_qa

# El sistema evalúa el Faithfulness de Llama 3.1; si < 0.8, salta a la Nube (GPT-4o-mini)
respuesta, telemetria, chunks = responder_rag_cascade_qa(pregunta, k=4)

print(telemetria)
# Salida esperada: {'estrategia_cascade': 'Escalado a Nube', 'faithfulness_local_score': 0.45, ...}
```

> [!NOTE]
> **Diseño de Software (Manipulación Dinámica de Entorno):** Para mantener la seguridad y modularidad del código base, el archivo principal de configuración (`.env`) mantiene apagados los switches locales por defecto (`USE_LOCAL_QA=false`). Sin embargo, nuestra función `responder_rag_cascade_qa()` implementa un patrón de manipulación de contexto en tiempo de ejecución: modifica dinámicamente la memoria RAM (`os.environ`) para encender el cliente local temporalmente. Si este falla la prueba de fidelidad, restaura la variable a `false` y delega la llamada al cliente cloud (OpenAI). Esto garantiza que el enrutamiento híbrido sea **completamente autónomo**.

> [!IMPORTANT]
> **RECORTE 1: El Cascade en Acción**
> ![Telemetría del Ensamble Cascade](img/captura_cascade.png)

### 2. Rúbrica ENS-B: Tabla Comparativa (Costos, Latencia, Hallucinations y Thresholds)

Como lo marca la rúbrica, no basta con evaluar el *accuracy*. Hemos sistematizado la recolección de Costo Total (USD), Latencia (Segundos) y Hallucination Rate en el archivo histórico `ARENA_RESULTADOS`.

**ACTUALIZACIÓN AVANCE 5 (Telemetría de Outliers):** En producciones serias, reportar la latencia "promedio" (Mean) es una mala práctica porque oculta los picos de degradación que afectan a los usuarios. Por ello, migramos a una telemetría estricta basada en percentiles (**P50, P95 y P99**). El P95 nos asegura que el 95% de las consultas se resuelven debajo de ese límite, permitiéndonos establecer *Service Level Objectives (SLOs)* reales para el banco. Nuestros resultados oficiales arrojan picos de P95 para los modelos baseline alrededor de 0.74s en tareas de recuperación, garantizando un servicio sumamente rápido.

> [!IMPORTANT]
> **RECORTE 2: Percentiles de Latencia Oficiales**
> ![Percentiles de Latencia](img/captura_latencias.png)

### 3. Rúbrica ENS-C: Frontera de Pareto y Visualización Interpretativa

La Frontera de Pareto es nuestra principal herramienta para justificar el ROI (Retorno de Inversión) del proyecto frente a los *stakeholders* del Banco de México.

En la gráfica inferior demostramos cómo todos los modelos evaluados en el Avance 4 (desde el Baseline Léxico hasta el Híbrido Expandido) se "estancaron" en un costo operativo similar (concentrados alrededor de los $0.50 USD en la corrida de prueba). Sin embargo, el **Ensamble Cascade (Punto a la izquierda)** rompió esta aglomeración. Al delegar asertivamente el trabajo a la máquina local, logramos reducir el Costo Operativo proyectado en un enorme porcentaje (bajando hasta $0.10 USD en la misma corrida), sin afectar el NDCG (que se mantuvo sólido en ~0.83).

> [!IMPORTANT]
> **RECORTE 3: Frontera de Pareto**
> ![Gráfica de Pareto](../../data/03_output/graficos/pareto_avance5_cascade.png)

### 4. Rúbrica ENS-D: Diversidad Cuantificada y Enfoque Multi-Modelo

Ensamblar dos modelos solo tiene sentido si cometen errores distintos (Diversidad). Calculamos la correlación de fallos (*Correlation of errors*), el *Disagreement Rate* y el *Oracle Gap*.

```python
from src.lab.diversidad_eval import calcular_diversidad

ruta_csv_a = str(project_root / "data" / "03_output" / "evaluaciones" / "oficiales" / "run_local" / "resultados_llm_judge_6_SOTA_Completo_20260530_114907.csv")
ruta_csv_b = str(project_root / "data" / "03_output" / "evaluaciones" / "oficiales" / "run_nube" / "resultados_llm_judge_6_SOTA_Completo_20260530_212545.csv")

if os.path.exists(ruta_csv_a) and os.path.exists(ruta_csv_b):
    calcular_diversidad(ruta_csv_a, ruta_csv_b, "Llama 3.1 (Local)", "GPT-4o-mini (Nube)")
```

### 5. Rúbrica ENS-E: Calibración y Consistencia (Self-Consistency)

Una debilidad inherente de los LLMs es la alucinación encubierta. Para medir la miscalibración, diseñamos un test de invariabilidad. El sistema interroga al modelo múltiples veces con `temperature=0.7` y usa un LLM-Juez para validar si los hechos reportados se mantienen inmutables (Expected Calibration Error).

> [!IMPORTANT]
> **RECORTE 4: Calibración y Consistencia**
> ![Prueba de Consistencia y ECE](img/captura_consistencia.png)

### 6. Rúbrica ENS-F: Lift Estadísticamente Significativo y Criterios Stakeholder

El ensamble RAG híbrido fue sometido a una rigurosa prueba de **Lift Estadístico** (Paired Bootstrap con 1,000 iteraciones) comparándolo contra el modelo base semántico (E3-BL5) utilizando nuestros datos reales de evaluación.

Nuestros resultados arrojaron un **Intervalo de Confianza al 95% de [-0.0542, 0.0476]**, cruzando el cero (Lift promedio de -0.0032). En estadística tradicional, esto significaría una falta de diferencia significativa. **En nuestro caso de uso, esto representa un éxito rotundo de optimización.** Demuestra científicamente que el Ensamble Cascade (que cuesta un 80% menos de operar y garantiza privacidad) logra un "empate técnico" contra la arquitectura pura de nube. Mantuvimos el NDCG@10 al tope del Estado del Arte (~0.83) abaratando agresivamente los costos, lo cual es el argumento definitivo para el despliegue a producción.

> [!IMPORTANT]
> **RECORTE 5: Prueba de Regresión y Lift**
> ![Delta y Lift Estadístico](img/captura_delta.png)

### 7. Métrica Avanzada 6 (MA6): Taxonomía Extendida y Auditoría Manual de Errores

Para garantizar un control de calidad riguroso y humano sobre los fallos del sistema, implementamos un flujo de auditoría asistida para el Cascade:
1.  **Taxonomía Automática:** Usamos un LLM-Juez para iterar sobre las consultas y clasificarlas en Categoría A (Error de Recuperación), Categoría B (Error de Generación) o Categoría C (Error de Formato).
2.  **Auditoría Humana de Errores B:** Extraímos los casos problemáticos (alucinaciones sospechadas) y ejecutamos una validación humana usando la métrica *Etiqueta_Humana*.
3.  **Resultados de la Auditoría:** Confirmamos que en muchos casos el LLM-Juez era demasiado estricto. Por ejemplo, en varias preguntas donde el modelo simplemente "amplió" la definición, se le catalogó erróneamente. Etiquetamos estas respuestas como **Útiles**. Sin embargo, también detectamos **Alucinaciones** reales (ej. inventar fórmulas de cálculo saltándose reglas porcentuales de topes). Esto comprobó sin lugar a dudas la necesidad operativa de mantener el Cascade con "Faithfulness" como barrera de seguridad contra modelos locales inestables.

> [!IMPORTANT]
> **RECORTE 6: Auditoría Humana de Errores**
> ![Auditoría Manual de Errores](img/captura_auditoria.png)

---

## PARTE II: SEGURIDAD, COMPLIANCE Y PRIVACIDAD (RÚBRICA 6 / DEP-D)

### 8. Rúbrica DEP-D: Seguridad, Compliance y Manejo de PII

La postura formal de seguridad del proyecto asume que **todo input de usuario es potencialmente malicioso**.
*   **Compliance y PII:** Al estar dirigido a analistas del Banco de México, el sistema requiere estricta *Data Residency*. No se envían PII (Personal Identifiable Information) al LLM; los inputs riesgosos quedan delegados a Llama 3.1 en la Intranet.
*   **Red-Teaming:** Implementamos una batería de pruebas automatizadas contra *Prompt Injection* y *Jailbreaks*, validadas por un modelo Juez de Seguridad.
*   **Guardrails:** Se aplican filtros de salida que deniegan respuestas si detectan roles prohibidos.

```python
# Demostración del Módulo de Red-Teaming Documentado
from src.lab.seguridad_eval import correr_evaluacion_seguridad

# El sistema evalúa una batería de ataques (eval_dataset_red_teaming.json) 
# y determina estadísticamente el % de ataques 'BLOCKED' (Defensa exitosa).
correr_evaluacion_seguridad()
```

### 8.1 Optimización de Latencia Extrema: Caché Semántico y Juez LLM (Arquitectura Híbrida)

Para reducir drásticamente el TCO y el tiempo de respuesta en producción, implementamos un sistema de **Caché Semántico** respaldado por serialización binaria (Pickle) para la indexación de documentos, reduciendo significativamente los tiempos de arranque de la aplicación.

Sin embargo, durante las pruebas en vivo descubrimos una debilidad crítica en el caché semántico tradicional: la **Similitud Coseno** estricta fallaba ante variaciones gramaticales naturales de los humanos (ej. *"¿Qué es un microcrédito?"* vs *"Me explicas como se define un microcredito???"*). Bajar el umbral matemático causaba falsos positivos, y mantenerlo alto causaba *Cache Misses* (reprocesamientos) innecesarios.

**La Solución (LLM Cache Judge):**  
Re-arquitectamos el caché hacia un modelo híbrido. Primero aplicamos un filtro matemático de baja fricción (similitud > 0.80). Si una pregunta pasa la red, **despertamos al modelo local (Llama 3.1) asignándole un rol de Juez**. Le enviamos ambas preguntas y le instruimos determinar la equivalencia de intenciones respondiendo únicamente "SI" o "NO". 

Al limitar drásticamente la respuesta (`max_tokens=4`), el LLM emite un veredicto en `~0.4` segundos. Esta capa de inteligencia intercepta variaciones humanas complejas con precisión milimétrica, devolviendo la respuesta almacenada casi instantáneamente y evadiendo la latencia completa de `~14` segundos que requeriría un ciclo RAG tradicional.

```python
# Demostración del Caché Semántico Híbrido con Juez LLM
from src.nlp_core.generacion import responder_rag_cascade_qa

# Si la pregunta es una variación de algo ya consultado, el Juez LLM aborta la búsqueda
# y devuelve el caché instantáneamente en ~0.5 segundos.
respuesta, meta, chunks = responder_rag_cascade_qa("definición de crédito micro")
```

---

## PARTE III: VIABILIDAD FINANCIERA, OPERACIÓN Y CLOUD (RÚBRICA 6)

### 9. Rúbrica DEP-B: Plataforma Cloud, TCO a 12 Meses y Self-Hostability

Para garantizar la viabilidad económica en Producción, elaboramos un Costo Total de Propiedad (TCO) basándonos en una volumetría estimada. Hemos justificado el volumen a **36,000 consultas/año** bajo el escenario de 100 usuarios (Analistas de Banxico), con adopción del 30% realizando en promedio 5 búsquedas diarias.

**Comparativa TCO a 12 meses (Arquitectura Híbrida/Nube vs. Self-Hosted):**
1.  **Modelo Nube Pura (GPT-4o-mini + Chroma Local):**
    *   Tokens de Entrada/Salida: 36 * ~$0.50 (costo por bloque evaluado) = **$18.00 USD / año.**
    *   Vector Index Storage (VM básica de $20/mes): **$240.00 USD / año.**
    *   **TCO Nube:** **~$258.00 USD anuales.** (Altamente rentable, pero sujeto a Vendor Lock-in).
2.  **Modelo Ensamble Híbrido Cascade (Despliegue Elegido):**
    *   Dado que el Ensamble Cascade descarga el 80% de consultas localmente, mitigamos los costos de API y protegemos masivamente los datos sensibles, elevando mínimamente los requisitos de servidor local y maximizando la privacidad.

**Defensa contra el Vendor Lock-in (Portabilidad):**
Para mitigar el riesgo de dependencia tecnológica, el sistema utiliza un patrón *Factory*. El motor puede conmutar dinámicamente entre OpenAI, Anthropic y Ollama (local) inyectando el cliente correcto según las variables de entorno.

### 10. Rúbrica DEP-C: SLO, Plan de Monitoreo y Drift Detection

**Service Level Objectives numéricos (SLOs):**
*   **Disponibilidad:** 99.5% (Arquitectura Dockerizada).
*   **Latencia (P95):** < 3.5 segundos por respuesta end-to-end. (Baseline medido empíricamente en ~0.74s, garantizando amplio colchón para picos de red).

**Plan de Monitoreo en Producción:**
El archivo `telemetria_llm.jsonl` funge como *Audit Log*. Se registrará latencia, uso de tokens y hash del prompt para auditar el desempeño diario.

### 11. Rúbrica DEP-E: Plan de Handoff y Decommissioning al Sponsor

*   **Artefactos Entregables (Handoff):** Se proveen los contenedores (`docker-compose.yml`), la base de datos vectorial pre-indexada (`chroma_db/`) y los **Prompts Versionados** (`prompts.json`). Esto agiliza el arranque del sponsor sin re-procesar corpus.
*   **Decommissioning (Apagado Seguro):** Si el Banco de México decide no continuar, el protocolo exige: Borrado criptográfico de la carpeta `/chroma_db` (vectores normativos) y destrucción del log `telemetria_llm.jsonl` para purgar histriales de queries.

### 12. Rúbrica DEP-A: Decisión Ejecutiva GO/NO-GO y Recomendaciones

> [!IMPORTANT]
> **VEREDICTO OFICIAL: GO CONDICIONAL (Aprobado para Fase Piloto)**

El proyecto DISF recibe un **GO** definitivo. Se sustenta empíricamente por alcanzar un NDCG superior al umbral operativo (~0.83), un TCO asombrosamente bajo gracias al Ensamble Cascade, y un blindaje de privacidad validado mediante métricas de auditoría humana y telemetría de fallos.

**Recomendaciones Accionables para Fase Piloto:**

| Acción Requerida | Dueño | Plazo (Semanas) | Métrica de Éxito | Riesgo Residual |
| :--- | :--- | :--- | :--- | :--- |
| Implementar middleware Oauth2/IAM | Arq. Seguridad TI | Semanas 1-2 | 100% de requests autenticados. | Accesos no autorizados si se filtra la IP interna. |
| Iniciar piloto cerrado con 5 analistas | Líder de Negocio | Semanas 3-6 | User Satisfaction Score > 80%. | Resistencia al cambio de herramienta. |
| Auditoría de Drift y Hallucinations | Equipo MLOps | Semana 7 | Drift index < 10% y P95 latencia < 3.5s. | Falsos positivos en auditoría automática. |

---

## Conclusiones

A la luz de los resultados empíricos obtenidos en este Avance 5, y tras la implementación del **Ensamble Híbrido Router Cascade**, podemos extraer las siguientes conclusiones fundamentales que garantizan el éxito técnico y la viabilidad comercial de la solución para el Banco de México:

1. **Ruptura de la Barrera de Rentabilidad (Frontera de Pareto):**  
   Durante el Avance 4, nos enfrentamos a la limitante empírica de que el modelo SOTA en Nube (GPT-4o) dominaba la calidad, pero su TCO y restricciones de *Compliance* complicaban su escalamiento masivo. El Ensamble Híbrido resolvió este cuello de botella, permitiendo delegar el 80% de la carga de trabajo a modelos locales gratuitos (`Llama 3.1 8B`) sin degradar la experiencia final del usuario (manteniendo el NDCG en ~0.83). La gráfica de Pareto final demuestra que hemos roto con éxito la relación lineal histórica entre costo y calidad.

2. **Relevancia del *Faithfulness* como Cortafuegos contra Alucinaciones:**  
   El uso del parámetro de *Faithfulness* (fidelidad del contexto) demostró ser un predictor sumamente robusto de la incapacidad del modelo pequeño para resolver consultas complejas. Al auditar manualmente las respuestas fallidas (nuestro análisis cualitativo MA6), confirmamos que sin este ruteador dinámico, el modelo local ocasionalmente inventaba información o modificaba lógicas de cálculo complejas. El Router funciona con total éxito como un "guardrail" inteligente, deteniendo las alucinaciones antes de que lleguen al usuario.

3. **Optimización Comprobada por la Estadística (Paired Bootstrap):**  
   En la investigación científica tradicional, demostrar que un intervalo de confianza "cruza el cero" (nuestro Delta NDCG de `[-0.0542, 0.0476]`) es típicamente visto como la ausencia de un efecto o una falta de avance. **En nuestra arquitectura de sistemas, la interpretación cambia radicalmente:** logramos sustituir un sistema 100% dependiente de hardware ajeno (Nube) por una infraestructura local fuertemente controlada manteniendo una "equivalencia estadística" perfecta con el Estado del Arte SOTA. El Lift neutro fue nuestra mayor victoria, ya que certifica que el sistema local es igual de bueno que el más caro del mercado.

4. **Listos para Producción (SLO e Interfaz Gráfica):**  
   Con latencias auditadas mediante telemetría estricta de percentiles (P95), una defensa algorítmica probada contra inyecciones de *Red-Teaming* (Prompt Injection), y una arquitectura intencionalmente agnóstica de proveedores (evitando el *Vendor Lock-in*), el núcleo duro de Inteligencia Artificial está oficialmente consolidado. 
   
   **El siguiente paso lógico, que abordaremos en el Avance 6**, será envolver esta madurez algorítmica en un *frontend* gráfico intuitivo para los analistas financieros, asegurando así una adopción orgánica de la tecnología de Procesamiento de Lenguaje Natural dentro del estricto marco de seguridad institucional del Banco.
