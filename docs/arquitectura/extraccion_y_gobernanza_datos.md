# Arquitectura: Extracción Estructurada y Gobierno de Datos (Data Provenance)

Este documento explora la arquitectura implementada en el proyecto DISF para garantizar la trazabilidad de la información (Data Provenance) y automatizar la extracción de metadatos y esquemas de formularios a partir de documentos regulatorios masivos de Banxico, haciendo uso de salidas estructuradas (Pydantic/JSON) y Grandes Modelos de Lenguaje (LLMs).

A lo largo del proyecto, estas estrategias evolucionaron de simples propuestas conceptuales a **prototipos totalmente funcionales** integrados en la aplicación web principal.

---

## 1. Data Provenance y Gestión de Metadatos

### ¿Qué es Data Provenance?
**Data Provenance** (Procedencia o Linaje de los Datos) es un concepto crítico en *MLOps* y Arquitecturas RAG que se refiere al registro histórico y auditable del origen, transformaciones y flujo de los datos desde su creación hasta su consumo por el LLM.

En un sistema RAG institucional, la confianza en el modelo depende enteramente de la confianza en los datos recuperados. Si el LLM responde basándose en un documento, necesitamos saber:
1. ¿De dónde salió ese documento?
2. ¿Cuál es su versión?
3. ¿A qué dominio normativo ("Tema") pertenece?

Sin esta trazabilidad, el repositorio se convierte en un **"Data Swamp"** (un pantano de datos donde los vectores no tienen contexto ni auditabilidad).

### El Estándar de Oro: `manifest.yaml`
El patrón arquitectónico estándar implementado para gobernar estos metadatos es el uso de un **Metadata Manifest** en formato YAML (`manifest.yaml`). Este archivo actúa como la única fuente de verdad (*Single Source of Truth*) para inyectar metadatos en la base de datos vectorial (ChromaDB).

YAML es superior a herramientas como Excel porque:
- Es texto plano, lo que permite un control de versiones impecable en Git sin conflictos binarios.
- Soporta comentarios para documentar por qué se agregó un archivo.
- Es el estándar en pipelines modernos de MLOps (MLflow, DVC, Kubernetes).

### Escalabilidad de Metadatos (RBAC y Precisión)
El uso del `manifest.yaml` habilita funcionalidades críticas:
1. **Filtros de Seguridad (RBAC):** Permite inyectar un pre-filtro en ChromaDB para que el RAG solo recupere contexto autorizado para un rol específico (garantizando que el LLM nunca lea documentos de áreas restringidas).
2. **Precisión del Modelo:** Atributos como `tipo_documento` evitan que el LLM mezcle una ley estricta con un manual operativo informal, acotando la búsqueda al dominio exacto.
3. **Gobernanza:** Asegura que el sistema jamás responda usando leyes derogadas.

---

## 2. Casos de Uso Core Implementados

El principal cuello de botella operativo del `manifest.yaml` es tener que llenarlo manualmente leyendo cientos de documentos. Para solucionarlo (y para resolver otros retos de la DISF), se implementaron dos flujos de extracción automatizada, los cuales ya cuentan con su interfaz dedicada:

### Caso A: Autogeneración de Metadatos de Gobernanza
Para organizar el repositorio masivo, se automatizó el etiquetado de cada archivo antes de ingresarlo al motor RAG.
- **Entrada:** Cualquier documento crudo subido en memoria temporal.
- **Proceso:** El modelo actúa como "Bibliotecario", deduciendo las características del documento usando la clase Pydantic `MetadataDocumento`.
- **Salida:** Una "Ficha Técnica" en pantalla y un JSON guardado en disco listo para inyectarse automáticamente en el `manifest.yaml` maestro (Tema, Institución, Confidencialidad).

### Caso B: Generación de Formularios desde Normativa
En lugar de dedicar semanas deduciendo qué datos solicitar a los bancos, el modelo infiere el esquema de base de datos requerido.
- **Entrada:** Un extracto normativo subido temporalmente en memoria.
- **Proceso:** El modelo actúa como "Analista de Datos", identificando las variables de origen. Todo forzado mediante un esquema rígido en Pydantic (`FormularioEstructurado`).
- **Salida:** Una tabla renderizada en pantalla con "Nombre del Campo", "Tipo de Dato", y un JSON estructural guardado en disco.

---

## 3. Paradigmas de Extracción: RAG vs Long-Context

**Actualmente, el sistema utiliza primordialmente el enfoque Long-Context para las pestañas de extracción estructurada**, dejando el RAG clásico exclusivamente para la consulta en el Chat.

### A. RAG (Retrieval-Augmented Generation) "Buscador Quirúrgico"
* **Uso Ideal:** Para extraer información específica de un mar de documentos (ej. el Chat preguntando sobre reglas de crédito a la CUB completa).
* **Desventaja en Extracción Estructural:** Si un campo de un formulario o el tema de un documento están esparcidos en varias páginas, el motor de recuperación semántica del RAG podría omitir un fragmento crucial, dejando la ficha técnica incompleta.

### B. Long-Context Extraction (La Estrategia Implementada)
* **Uso Ideal:** Para extraer esquemas o metadatos completos de un documento particular cargado en memoria.
* **Cómo Funciona:** Los modelos modernos (como GPT-4o) tienen una ventana de contexto de 128,000 tokens (aprox. 300 páginas). En `generacion.py`, **no filtramos por base de datos vectorial**. Pasamos el documento entero al *Prompt* del modelo.
* **Ventaja Insuperable:** El modelo tiene visión panorámica. Entiende referencias cruzadas y garantiza no omitir campos. Al eliminar a ChromaDB de este flujo, **la precisión de la extracción estructural aumenta radicalmente**.

---

## 4. Anatomía del Prototipo Construido

Para lograr fiabilidad en producción, se construyeron:
1. **Esquemas Pydantic (`schemas.py`):** Clases estrictamente tipadas que obligan al LLM a responder con claves JSON exactas, previniendo alucinaciones estructurales.
2. **Prompts Especializados (`prompts.json`):** Instrucciones de sistema sin formato de chat, que otorgan el "Rol" adecuado.
3. **Flujo de Interfaz Directa (`index.html` y `main.js`):** El usuario interactúa a través de un panel de "Control de Extracción" de un solo clic, sin cajas de chat, aislando la experiencia hacia la generación de datos estructurados.

---

## 5. Propuestas de Arquitectura a Futuro (Enterprise-grade)

A medida que el sistema escale hacia regulaciones de miles de páginas que superen el límite *Long-Context*, se recomienda migrar hacia:

### A. Arquitectura "Map-Reduce" Jerárquica
* **Map:** Pre-procesar el documento dividiéndolo por "Capítulos" y hacer múltiples extracciones en paralelo.
* **Reduce:** Un LLM consolidador elimina duplicados y entrega un único esquema unificado.

### B. Flujo Multi-Agente (Agentic Workflow)
Varios LLMs asumen diferentes roles automatizados (Agente Extractor, Agente Validador Crítico, Agente Consolidador) para verificar tipos de datos lógicos y mitigar la necesidad de revisión humana constante.

### C. "Human-in-the-Loop" (HITL) Interactivo
Actualmente el sistema arroja el resultado final. En el futuro:
* El LLM asignará una bandera de "Baja Confianza" a los campos dudosos.
* La interfaz resaltará esas filas.
* El usuario revisará y modificará *únicamente* esas filas directamente en la web antes de guardar la estructura en la base de datos maestra.
