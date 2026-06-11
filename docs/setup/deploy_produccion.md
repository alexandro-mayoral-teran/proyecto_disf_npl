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

## 3. Instrucciones Paso a Paso (Workflow Automatizado)

Dado que usaremos un esquema de Monorepo, **no necesitas copiar archivos manualmente de una carpeta a otra en tu computadora**. Haremos todo directamente desde tu repositorio actual (`proyecto_disf_npl`) usando ramas de Git automatizadas.

### Configuración Inicial (Solo se hace una vez)
1. Ve a [huggingface.co/spaces](https://huggingface.co/spaces) y crea **dos Spaces** nuevos:
   - **`rag-cub-webapp`**: Elige la licencia `mit`, SDK **Docker** (plantilla Blank) y Hardware Free.
   - **`rag-cub-dashboard`**: Elige la licencia `mit`, SDK **Streamlit** y Hardware Free.
2. Ve a la pestaña "Settings" del **primer Space (webapp)**, busca "Variables and secrets", y agrega un **New Secret** (`Name: OPENAI_API_KEY`, `Value: sk-...`).
3. Repite el paso 2 para el **segundo Space (dashboard)**. ¡Nunca subas el archivo `.env` directamente!
4. Abre tu terminal local en la raíz del proyecto y vincula ambos remotos (reemplaza `TU_USUARIO` por tu nombre en HF):
   ```bash
   git remote add space-api https://huggingface.co/spaces/TU_USUARIO/rag-cub-webapp
   git remote add space-dashboard https://huggingface.co/spaces/TU_USUARIO/rag-cub-dashboard
   ```
5. *(Requisito MLOps)*: Asegúrate de haber migrado tu base de datos y archivos binarios a Git LFS para que no sean rechazados por el límite de 10 MB:
   ```bash
   git lfs install
   git lfs migrate import --include="*.bin,*.sqlite3,*.docx,*.pdf,*.pickle,*.png,*.pkl" --everything
   ```

### Despliegue y Actualizaciones (Script `deploy_spaces.ps1`)

Hugging Face lee el archivo `README.md` para saber qué tecnología arrancar, pero no podemos poner la configuración de Docker y Streamlit en el mismo archivo simultáneamente.

Para solucionar este reto de monorepo sin ensuciar tu rama principal, hemos creado el script **`deploy_spaces.ps1`**. Este script clona tu código en dos ramas invisibles, inyecta la configuración correcta en cada una, y las sube a sus respectivos servidores.

Cada vez que programes algo nuevo y quieras **actualizar producción**, solo sigue estos 2 pasos:

1. **Guarda tus cambios localmente:**
   ```bash
   git add .
   git commit -m "Escribe aquí lo que mejoraste"
   ```
2. **Ejecuta el script de despliegue:**
   ```bash
   ./deploy_spaces.ps1
   ```
   > [!IMPORTANT]
   > **Autenticación:** Hugging Face te pedirá usuario y contraseña en la terminal. En la contraseña NO pongas tu clave normal, debes pegar un **Access Token** con permisos de **Write** (puedes crearlo en `Settings -> Access Tokens` de tu perfil de Hugging Face).

¡Y listo! El script subirá automáticamente el código a ambos servidores y tu aplicación web y tablero MLOps se actualizarán en vivo.

## 4. Troubleshooting y Lecciones Aprendidas (MLOps)

Durante el primer despliegue a producción en servidores en la nube como Hugging Face, es común enfrentar restricciones de seguridad y entorno. Si otro miembro del equipo necesita reproducir este despliegue, debe tener en cuenta lo siguiente:

### A. Restricciones de Archivos Binarios y Pesados (Git LFS)
Hugging Face Spaces bloqueará cualquier intento de `git push` si detecta:
1. Archivos mayores a 10 MB (ej. nuestra base de datos vectorial de ChromaDB).
2. Archivos puramente binarios, sin importar su tamaño (ej. `.docx`, `.pdf`, `.png`, `.pickle`, `.sqlite3`).

**Solución:** Siempre debemos forzar a que Git trate todas estas extensiones mediante Git Large File Storage (LFS) antes de subir el código:
```bash
git lfs migrate import --include="*.bin,*.sqlite3,*.docx,*.pdf,*.pickle,*.png,*.pkl" --everything
```

### B. El Script `deploy_spaces.ps1` en diferentes Terminales
Si intentas correr `./deploy_spaces.ps1` en una terminal de Linux (como **Git Bash** o **MINGW64**), el sistema arrojará múltiples errores de sintaxis (`command not found`). Esto ocurre porque el script está diseñado para el motor nativo de Windows (PowerShell).

**Solución:** Ejecutar explícitamente el script invocando a PowerShell y saltando las políticas de restricción de Windows:
```bash
powershell -ExecutionPolicy Bypass -File .\deploy_spaces.ps1
```

### C. Codificación de Emojis en YAML y Versiones de Python
El archivo `README.md` requiere un encabezado YAML. Se descubrieron dos retos:
1. **Emojis corruptos:** Al inyectar texto mediante PowerShell en Windows, el "Encoding UTF-8" por defecto corrompe los caracteres especiales (ej. transformándolos en `ðŸš€`). Hugging Face rechaza el código marcando un `YAML metadata verification error`. *Solución:* No usar emojis en scripts automatizados.
2. **Versiones de Python:** Al definir `python_version: 3.10` sin comillas, el motor de YAML eliminó el cero (leyendo 3.1), lo cual causó que el contenedor de Docker fallara. *Solución:* Usar comillas `"3.10"` explícitamente.

### D. Ausencia de Modelos Locales en la Nube (UnboundLocalError)
El sistema "RAG Cascade" estaba diseñado para probar primero con un modelo local (Ollama). Al subir el código a la nube, Ollama no existe, por lo que la red fallaba tan rápido que ciertas variables de telemetría nunca llegaban a inicializarse, causando un "Error Interno del Servidor".

**Solución:** El código fuente fue parcheado para inicializar preventivamente estas variables (`meta_local = {}`) y atrapar correctamente fallas de red catastróficas, garantizando que el "Plan B" (servidores de OpenAI en la nube) siempre entre al rescate.

## 5. Privacidad y Consumo de Tokens (OpenAI)

Tus Spaces en Hugging Face están configurados como **Public** (Públicos). 

**¿Esto es un riesgo para mis tokens de OpenAI?**
No. En la práctica, el ecosistema funciona bajo "Seguridad por Oscuridad". Nadie en internet encontrará tu aplicación a menos que tú les compartas la liga exacta (`huggingface.co/spaces/TU_USUARIO/...`). 
Dejarlo como *Público* es **altamente recomendado para tu evaluación académica**, ya que permite que todo tu jurado o asesores puedan entrar a probarlo inmediatamente con solo dar clic en la liga, sin necesidad de crear cuentas en Hugging Face ni pedirte accesos.

**¿Cómo protejo mi tarjeta cuando me evalúen?**
En cuanto termine tu evaluación, simplemente ve a la pestaña *Settings* de tus Spaces y elimina el *Secret* llamado `OPENAI_API_KEY` (o haz clic en "Pause Space"). Esto "apagará" el cerebro de la aplicación permanentemente, garantizando que ya no consuma un solo centavo de tu cuenta de OpenAI.
