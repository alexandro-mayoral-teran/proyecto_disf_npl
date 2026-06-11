# Manual de Despliegue en Producción (Demo)

Este documento detalla las opciones, la decisión arquitectónica y las instrucciones paso a paso para desplegar el "Proyecto Integrador RAG Normativo" en la nube para demostraciones públicas.

---

## 1. El Reto de Nuestra Arquitectura Multi-Servicio

A diferencia de proyectos escolares simples, nuestro sistema tiene una arquitectura profesional dividida en dos grandes bloques:
1. **La App Web Principal (El Producto):** Un backend en FastAPI (`api/main_api.py`) que sirve nuestra base de datos vectorial (ChromaDB) y, al mismo tiempo, sirve los archivos estáticos del frontend en Vanilla JS (`app/`).
2. **El Dashboard MLOps (La Supervisión):** Una aplicación separada en Streamlit (`dashboard/app_evaluaciones.py`) utilizada por los ingenieros para monitorear latencias, costos y la trazabilidad (Caja Blanca).

Para desplegar esto gratis, evaluamos tres alternativas:
- **Streamlit Cloud:** Solo soporta Streamlit. Borra los archivos (sin persistencia) y sufre de *Cold Starts* (se duerme).
- **Render:** Permite Web Services (FastAPI), pero su capa gratuita solo da 512MB de RAM, lo cual provocará que nuestro ChromaDB y los modelos de Embeddings colapsen por falta de memoria (Out of Memory).
- **Hugging Face (HF) Spaces:** Nos regala 16 GB de RAM, 2 vCPU y **Persistencia de Disco** (nuestro caché y base de datos vectorial sobrevivirán reinicios). 

---

## 2. Decisión Arquitectónica: Estrategia de Dos "Spaces" (Microservicios)

Dado que HF Spaces solo expone un puerto público a la vez, la solución más elegante y robusta es implementar un patrón de **Microservicios** creando **DOS Spaces gratuitos** en Hugging Face:

1. **Space A (App Web FastAPI):** Usaremos el SDK "Docker" de Hugging Face para levantar `uvicorn api.main_api:app`. Esto nos dará 16GB de RAM dedicados exclusivamente a buscar vectores y responder al usuario en la web.
2. **Space B (Dashboard Streamlit):** Usaremos el SDK "Streamlit" para correr `dashboard/app_evaluaciones.py`, con otros 16GB de RAM dedicados a leer las gráficas y telemetría.

*(Opcional: Si ambas apps necesitan leer el mismo `semantic_cache.pkl` físico, se puede montar un disco persistente compartido en HF, o simplemente mantener el Dashboard apuntando a copias estáticas de los logs para la demo).*

### 2.1 Estrategia de Control de Versiones (Git): El Monorepo
Es fundamental aclarar que **NO** se crearán repositorios separados para cada microservicio. Adoptaremos una estrategia de **Monorepo** (usando el mismo repositorio actual `proyecto_disf_npl`). 
¿Por qué? Porque tanto el backend (API) como el dashboard (Streamlit) comparten dependencias críticas: la carpeta `src/nlp_core/` (lógica de RAG) y la carpeta `data/`. Separar el proyecto obligaría a duplicar código.
Hugging Face es 100% compatible con monorepos: al configurar el Dockerfile del Space A le indicaremos que solo levante la API, y en el YAML del Space B le diremos que solo lea la carpeta del dashboard. De esta forma, un solo `git push` a tu repositorio principal puede sincronizar y actualizar ambos microservicios a la vez.

### 2.2 Archivos y Carpetas a Subir
Para levantar correctamente ambos *Spaces*, deberás subir (mediante Git) la siguiente estructura base idéntica a los dos repositorios en Hugging Face:
- `api/` (Contiene `main_api.py` y las rutas del backend).
- `app/` (Contiene el HTML/JS/CSS del frontend interactivo).
- `dashboard/` (Contiene `app_evaluaciones.py` para Streamlit).
- `src/` (Contiene el núcleo de NLP y RAG compartido).
- `data/` (Muy importante: aquí debe ir tu base vectorial ChromaDB y archivos estáticos de telemetría).
- `requirements.txt` (Las dependencias del entorno).

### 2.3 Telemetría en la Demostración (Aislamiento de Spaces)
Dado que estamos usando dos Spaces en la capa gratuita, **sus discos duros están aislados**. Esto significa que si haces una pregunta en la App Web (Space A), la telemetría se guardará en ese disco y **NO** se reflejará automáticamente en el Dashboard (Space B) en tiempo real.
Para mantener el despliegue simple y sin fricciones durante la demostración (sin usar bases de datos externas compartidas):
- Ambos repositorios compartirán un archivo `.jsonl` de telemetría **estático**, subido manualmente y pre-cargado en el repositorio (ej. dentro de la carpeta `data/` o en la raíz).
- De esta forma, el Dashboard de Streamlit mostrará un historial robusto de pruebas, latencias reales (P50/P95/P99) y la Frontera de Pareto estática sin errores; mientras el jurado o los analistas pueden usar la App Web en vivo de forma independiente.

---

## 3. Instrucciones Paso a Paso (Workflow con Git)

Dado que usaremos un esquema de Monorepo, **no necesitas copiar archivos manualmente de una carpeta a otra en tu computadora**. Haremos todo directamente desde tu repositorio actual (`proyecto_disf_npl`) agregando múltiples "remotos" de Git.

### Despliegue 1: La App Web Principal (FastAPI + Vanilla JS)
1. Ve a [huggingface.co/spaces](https://huggingface.co/spaces) y crea un nuevo Space.
   - **Space name**: `rag-cub-webapp`
   - **License**: `mit` (Es la estándar para proyectos académicos y demostraciones).
   - **Select the Space SDK**: Elige **Docker** (plantilla Blank).
   - **Hardware**: Free (CPU Basic).
2. En la raíz de tu proyecto local, crea un archivo llamado `Dockerfile` con el siguiente código para decirle a HF cómo correr tu FastAPI:
   ```dockerfile
   FROM python:3.10
   WORKDIR /code
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   # HF Spaces expone el puerto 7860 por defecto
   CMD ["uvicorn", "api.main_api:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
3. En la pestaña "Settings" del Space recién creado en la web, busca la sección "Variables and secrets", y agrega un **New Secret** (`Name: OPENAI_API_KEY`, `Value: sk-...`). ¡Nunca subas el archivo `.env` directamente!
4. Abre tu terminal en la carpeta de tu proyecto local y vincula tu repositorio con el Space A ejecutando:
   ```bash
   git remote add space-api https://huggingface.co/spaces/TU_USUARIO/rag-cub-webapp
   ```
5. Haz commit de tu nuevo `Dockerfile` y empuja todo tu código al Space:
   ```bash
   git add Dockerfile
   git commit -m "Agregar Dockerfile para App Web"
   git push space-api main
   ```
   *(Nota: Te pedirá tu usuario de HF y un Access Token como contraseña).*
6. ¡Tu App Web estará construyéndose y en unos minutos estará viva!

### Despliegue 2: El Dashboard MLOps (Streamlit)
1. Ve nuevamente a [huggingface.co/spaces](https://huggingface.co/spaces) y crea un segundo Space.
   - **Space name**: `rag-cub-dashboard`
   - **Select the Space SDK**: Elige **Streamlit**.
2. En la raíz de tu proyecto local, abre tu archivo `README.md` y asegúrate de inyectar al **principio del archivo** este bloque YAML (HF lo lee para saber qué archivo de Python debe arrancar):
   ```yaml
   ---
   title: Dashboard MLOps RAG
   emoji: 📊
   colorFrom: blue
   colorTo: indigo
   sdk: streamlit
   sdk_version: 1.25.0
   app_file: dashboard/app_evaluaciones.py
   pinned: false
   ---
   ```
3. En tu terminal, vincula tu repositorio local con este segundo Space:
   ```bash
   git remote add space-dashboard https://huggingface.co/spaces/TU_USUARIO/rag-cub-dashboard
   ```
4. Haz commit de tu `README.md` actualizado y empuja tu código al Space B:
   ```bash
   git add README.md
   git commit -m "Configurar YAML para Streamlit"
   git push space-dashboard main
   ```
5. ¡Listo! Tendrás tu panel de métricas corriendo en otra URL independiente. 

Ambos Spaces comparten la misma base de código desde tu computadora, pero al hacer el `git push` a un remoto diferente, se ejecutan de manera aislada en Hugging Face cumpliendo sus propósitos específicos.
