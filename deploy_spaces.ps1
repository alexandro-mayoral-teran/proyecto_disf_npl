Write-Host "=== Actualizando Hugging Face Spaces ==="
git checkout -f main

# Space A: API
Write-Host "-> Preparando Web App (Docker)..."
git branch -D deploy-api 2>$null
git checkout --orphan deploy-api
$apiYaml = @"
---
title: RAG CUB Webapp
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

"@
$content = Get-Content README.md -Raw
Set-Content README.md -Value ($apiYaml + $content) -Encoding UTF8
git add .
# Excluir archivos del dashboard para no ocupar espacio en la API
git rm -r --cached "data/03_output/evaluaciones_*" 2>$null
git rm -r --cached "data/03_output/telemetria_llm.jsonl" 2>$null
git rm -r --cached "dashboard" 2>$null
git commit -m "Configurar metadata Docker"
git push -f space-api deploy-api:main

# Space B: Dashboard
Write-Host "-> Preparando Dashboard (Streamlit)..."
git checkout -f main
git branch -D deploy-dashboard 2>$null
git checkout --orphan deploy-dashboard
$dashYaml = @"
---
title: Dashboard MLOps RAG
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.25.0
python_version: "3.10"
app_file: dashboard/app_evaluaciones.py
pinned: false
---

"@
$content = Get-Content README.md -Raw
Set-Content README.md -Value ($dashYaml + $content) -Encoding UTF8
git add .
# Excluir archivos de la API para no ocupar espacio en el Dashboard
git rm -r --cached "data/03_output/chroma_db*" 2>$null
git rm -r --cached "data/03_output/semantic_cache*" 2>$null
git rm -r --cached "app" 2>$null
git rm -r --cached "api" 2>$null
git commit -m "Configurar metadata Streamlit"
git push -f space-dashboard deploy-dashboard:main

git checkout -f main
Write-Host "=== ¡Todos los despliegues finalizados! ==="
