// Referencias DOM - General
const latencyHud = document.getElementById('latency-hud');

// Referencias DOM - Tab 0: QA RAG
const qaForm = document.getElementById('qa-form');
const qaInput = document.getElementById('qa-input');
const qaHistory = document.getElementById('qa-history');
const btnQaSend = document.getElementById('btn-qa-send');

// Configuradores RAG
const configRetriever = document.getElementById('config-retriever');
const configExpansion = document.getElementById('config-expansion');
const configPost = document.getElementById('config-post');

// Referencias DOM - Tab 1: Extracción Formularios
const chatForm = document.getElementById('chat-form');
const queryInput = document.getElementById('query-input');
const chatHistory = document.getElementById('chat-history');
const resultContainer = document.getElementById('result-container');
const btnSend = document.getElementById('btn-send');

// Cambio de Pestañas
function switchTab(tabId, navElement) {
    // Quitar active de todos los links
    document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
    // Agregar active al clickeado
    navElement.classList.add('active');
    
    // Ocultar todos los tab panes
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    // Mostrar el seleccionado
    document.getElementById(tabId).classList.add('active');

    // Pestaña de evaluaciones eliminada (movida a Jupyter)
}

// Función Genérica para agregar mensajes al chat
function addChatMessage(message, sender, container) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('chat-message', sender);
    
    if (sender === 'bot') {
        // Usar marked para parsear Markdown a HTML
        msgDiv.innerHTML = marked.parse(message);
        
        // Usar KaTeX para renderizar matemáticas
        if (window.renderMathInElement) {
            renderMathInElement(msgDiv, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError: false
            });
        }
    } else {
        msgDiv.textContent = message;
    }
    
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    
    return msgDiv;
}

// ----------------------------------------------------
// LÓGICA: Pestaña 0 - Consulta Normativa (RAG Puro)
// ----------------------------------------------------

function renderQATelemetry(telemetry, context, config, parentMsgDiv) {
    const detailsEl = document.createElement('details');
    detailsEl.style.marginTop = '1rem';
    detailsEl.style.fontSize = '0.8rem';
    detailsEl.style.background = '#f8f9fa';
    detailsEl.style.border = '1px solid rgba(210, 223, 232, 0.8)';
    detailsEl.style.borderRadius = '6px';
    detailsEl.style.padding = '0.5rem';

    let html = `
        <summary style="font-weight: bold; cursor: pointer; color: var(--banxico-azul-institucional); outline: none;">
            🔍 Métricas y Contexto RAG (${config.base_retriever} + ${config.query_expansion})
        </summary>
        <div style="margin-top: 0.8rem; border-top: 1px solid #e2e8f0; padding-top: 0.5rem;">
            <div style="display: flex; gap: 1rem; margin-bottom: 0.8rem;">
                <div style="flex: 1; background: white; padding: 0.4rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-size: 0.7rem;">⚡ Tokens (Prompt/Comp)</span>
                    <div style="font-weight: bold; color: var(--banxico-azul-oscuro);">${telemetry.total_tokens} (${telemetry.prompt_tokens}/${telemetry.completion_tokens})</div>
                </div>
                <div style="flex: 1; background: white; padding: 0.4rem; border-radius: 4px; border: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-size: 0.7rem;">⏱️ Latencia Total</span>
                    <div style="font-weight: bold; color: var(--banxico-azul-oscuro);">${telemetry.latencia_total_seg}s (Busq: ${telemetry.latencia_busqueda_seg}s / LLM: ${telemetry.latencia_llm_seg}s)</div>
                </div>
            </div>
            <h4 style="font-size: 0.8rem; margin-bottom: 0.4rem; color: var(--banxico-azul-medio);">Fragmentos Extraídos (${context.length})</h4>
            <div style="display: flex; flex-direction: column; gap: 0.4rem; max-height: 250px; overflow-y: auto;">
    `;
    
    context.forEach((chunk, index) => {
        const sourceName = chunk.metadata.documento || "Documento desconocido";
        html += `
            <details style="background: white; border: 1px solid #e2e8f0; border-radius: 4px; padding: 0.4rem;">
                <summary style="font-weight: bold; font-size: 0.75rem; color: var(--banxico-azul-institucional); cursor: pointer;">
                    [${index + 1}] ${sourceName}
                </summary>
                <div style="font-size: 0.7rem; font-family: monospace; background: #f1f5f9; padding: 0.4rem; border-radius: 4px; margin-top: 0.4rem; white-space: pre-wrap;">${chunk.content}</div>
            </details>
        `;
    });
    
    html += `</div></div>`;
    detailsEl.innerHTML = html;
    parentMsgDiv.appendChild(detailsEl);
    
    // Auto scroll después de expandir
    detailsEl.addEventListener('toggle', (e) => {
        if(detailsEl.open) {
            parentMsgDiv.parentElement.scrollTop = parentMsgDiv.parentElement.scrollHeight;
        }
    });
}

qaForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = qaInput.value.trim();
    if (!query) return;

    // Obtener configuración del pipeline
    const config = {
        base_retriever: configRetriever.value,
        query_expansion: configExpansion.value,
        post_processing: configPost.value
    };

    addChatMessage(query, 'user', qaHistory);
    qaInput.value = '';
    
    const btnHtml = btnQaSend.innerHTML;
    btnQaSend.innerHTML = '<div class="loading-spinner" style="width: 20px; height: 20px; border: 2px solid white; border-top-color: transparent; border-radius: 50%; animation: loading-spin 1s linear infinite;"></div>';
    btnQaSend.disabled = true;
    latencyHud.innerHTML = `<i data-lucide="activity" class="icon-sm"></i> Latencia: calculando...`;
    
    // Crear un skeleton para el chat mientras responde
    const skeletonDiv = document.createElement('div');
    skeletonDiv.className = 'chat-message bot skeleton skeleton-text';
    skeletonDiv.style.width = '60%';
    qaHistory.appendChild(skeletonDiv);
    qaHistory.scrollTop = qaHistory.scrollHeight;

    try {
        const response = await fetch('/api/consulta_normativa', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: query, 
                top_k: 4,
                base_retriever: config.base_retriever,
                query_expansion: config.query_expansion,
                post_processing: config.post_processing
            })
        });

        const jsonResp = await response.json();
        
        // Quitar skeleton si existe
        if (qaHistory.contains(skeletonDiv)) {
            qaHistory.removeChild(skeletonDiv);
        }

        if (!response.ok) throw new Error(jsonResp.detail || "Error en el servidor");

        const textoMarkdown = jsonResp.data;
        const telemetry = jsonResp.telemetry;
        const context = jsonResp.context;

        const msgDiv = addChatMessage(textoMarkdown, 'bot', qaHistory);
        renderQATelemetry(telemetry, context, config, msgDiv);

        // Actualizar HUD
        latencyHud.innerHTML = `<i data-lucide="activity" class="icon-sm"></i> Latencia: ${telemetry.latencia_total_seg}s`;
        lucide.createIcons();
        
    } catch (error) {
        if (qaHistory.contains(skeletonDiv)) {
            qaHistory.removeChild(skeletonDiv);
        }
        addChatMessage("Ocurrió un error: " + error.message, 'bot', qaHistory);
    } finally {
        btnQaSend.innerHTML = btnHtml;
        btnQaSend.disabled = false;
        qaInput.focus();
    }
});


// ----------------------------------------------------
// LÓGICA: Pestaña 1 - Extracción Formularios (Pydantic)
// ----------------------------------------------------

function showTableSkeleton() {
    resultContainer.innerHTML = `
        <div class="skeleton skeleton-text" style="width: 60%"></div>
        <div class="skeleton skeleton-chart" style="height: 200px; margin-top: 1rem;"></div>
    `;
}

function renderResultTable(data) {
    if (!data.campos_formulario || data.campos_formulario.length === 0) {
        resultContainer.innerHTML = `<div class="callout callout-important">No se encontraron campos para esta consulta.</div>`;
        return;
    }

    let html = `<h4 style="color: var(--banxico-azul-institucional);">${data.nombre_formulario}</h4>`;
    html += `<table class="data-table">
                <thead>
                    <tr>
                        <th>Campo</th>
                        <th>Tipo Dato</th>
                        <th>Descripción</th>
                        <th>Fórmula</th>
                    </tr>
                </thead>
                <tbody>`;
    
    data.campos_formulario.forEach(campo => {
        let catalogoBadge = campo.es_catalogo ? `<br><span class="badge-catalogo">Catálogo: ${campo.nombre_catalogo_vinculado || 'Sí'}</span>` : '';
        let formulaText = campo.formula_calculo ? `<strong>${campo.formula_calculo}</strong>` : '-';
        
        html += `<tr>
                    <td><strong>${campo.nombre_campo}</strong>${catalogoBadge}</td>
                    <td>${campo.tipo_dato}</td>
                    <td>${campo.descripcion_funcional}</td>
                    <td>${formulaText}</td>
                 </tr>`;
    });
    
    html += `</tbody></table>`;

    if (data.ambiguedades_detectadas && data.ambiguedades_detectadas.length > 0) {
        html += `<div class="callout callout-important" style="margin-top: 1rem;">
                    <strong>⚠️ Ambigüedades Detectadas:</strong>
                    <ul style="margin: 0.5rem 0 0; padding-left: 1.5rem;">`;
        data.ambiguedades_detectadas.forEach(amb => {
            html += `<li>${amb}</li>`;
        });
        html += `</ul></div>`;
    }

    resultContainer.innerHTML = html;
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    addChatMessage(query, 'user', chatHistory);
    queryInput.value = '';
    btnSend.disabled = true;
    showTableSkeleton();
    addChatMessage("Ejecutando RAG y estructurando respuesta Pydantic...", 'bot', chatHistory);

    try {
        const response = await fetch('/api/extraer_formulario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, top_k: 4 })
        });

        const jsonResp = await response.json();

        if (!response.ok) throw new Error(jsonResp.detail || "Error en el servidor");

        const formResponse = jsonResp.data;
        const telemetry = jsonResp.telemetry;

        renderResultTable(formResponse);
        addChatMessage(`¡Listo! Extraje ${formResponse.campos_formulario.length} campos. Revisa la tabla de la derecha.`, 'bot', chatHistory);

        latencyHud.innerHTML = `<i data-lucide="activity" class="icon-sm"></i> Latencia: ${telemetry.latencia_total_seg}s`;
        lucide.createIcons();
        
    } catch (error) {
        addChatMessage("Ocurrió un error: " + error.message, 'bot', chatHistory);
        resultContainer.innerHTML = `<div class="callout callout-important">Error al consultar el modelo: ${error.message}</div>`;
    } finally {
        btnSend.disabled = false;
        queryInput.focus();
    }
});


// ----------------------------------------------------
// (El módulo de evaluaciones se trasladó a Jupyter Notebook)
// ----------------------------------------------------
