Write-Host "=== Actualizando Hugging Face Spaces ==="
git checkout main

# Space A: API
Write-Host "-> Preparando Web App (Docker)..."
git checkout -B deploy-api
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
git commit -am "Configurar metadata Docker"
git push -f space-api deploy-api:main

# Space B: Dashboard
Write-Host "-> Preparando Dashboard (Streamlit)..."
git checkout main
git checkout -B deploy-dashboard
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
git commit -am "Configurar metadata Streamlit"
git push -f space-dashboard deploy-dashboard:main

git checkout main
Write-Host "=== ¡Todos los despliegues finalizados! ==="
