// Referencias DOM - General
const latencyHud = document.getElementById('latency-hud');

// Referencias DOM - Tab 0: QA RAG
const qaForm = document.getElementById('qa-form');
const qaInput = document.getElementById('qa-input');
const qaHistory = document.getElementById('qa-history');
const btnQaSend = document.getElementById('btn-qa-send');

// Referencias DOM - Tab 1: Extracción Formularios
const chatForm = document.getElementById('chat-form');
const resultContainer = document.getElementById('result-container');
const btnSend = document.getElementById('btn-send');
const formStatus = document.getElementById('extraccion-form-status');

// Referencias DOM - Tab 2: Extracción Metadatos
const metaForm = document.getElementById('meta-form');
const btnMetaSend = document.getElementById('btn-meta-send');
const metaResultContainer = document.getElementById('meta-result-container');
const metaStatus = document.getElementById('extraccion-meta-status');

// Controles Globales (Efímero y Temas)
const temaSelect = document.getElementById('tema-select');
const efimeroFile = document.getElementById('efimero-file');
const btnUploadEfimero = document.getElementById('btn-upload-efimero');
const efimeroStatus = document.getElementById('efimero-status');
const soloEfimeroCheck = document.getElementById('solo-efimero-check');
const dbSelector = document.getElementById('db-selector');

let textosEfimerosGlobal = null;

btnUploadEfimero.addEventListener('click', async () => {
    const file = efimeroFile.files[0];
    if (!file) {
        efimeroStatus.textContent = "⚠️ Selecciona un archivo PDF/Word primero.";
        return;
    }
    
    efimeroStatus.textContent = "Procesando documento...";
    btnUploadEfimero.disabled = true;
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        const response = await fetch('/api/upload_efimero', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (response.ok && result.status === "success") {
            textosEfimerosGlobal = [result.markdown];
            efimeroStatus.textContent = "✅ ¡Cargado en Memoria RAM!";
        } else {
            efimeroStatus.textContent = "❌ Error: " + (result.detail || "Desconocido");
        }
    } catch (err) {
        efimeroStatus.textContent = "❌ Error de red al subir archivo.";
    } finally {
        btnUploadEfimero.disabled = false;
    }
});

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

function renderQATelemetry(telemetry, context, parentMsgDiv) {
    const detailsEl = document.createElement('details');
    detailsEl.style.marginTop = '1rem';
    detailsEl.style.fontSize = '0.8rem';
    detailsEl.style.background = '#f8f9fa';
    detailsEl.style.border = '1px solid rgba(210, 223, 232, 0.8)';
    detailsEl.style.borderRadius = '6px';
    detailsEl.style.padding = '0.5rem';

    let cascade_badge = telemetry.estrategia_cascade ? `<span style="background: var(--banxico-dorado); color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; margin-left: 5px;">${telemetry.estrategia_cascade}</span>` : '';

    let html = `
        <summary style="font-weight: bold; cursor: pointer; color: var(--banxico-azul-institucional); outline: none;">
            🔍 Métricas de Generación (Cascade) ${cascade_badge}
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
                tema: temaSelect.value,
                textos_efimeros: textosEfimerosGlobal,
                solo_efimero: soloEfimeroCheck.checked,
                db_folder: dbSelector ? dbSelector.value : "chroma_db"
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
        renderQATelemetry(telemetry, context, msgDiv);

        // Actualizar HUD
        latencyHud.innerHTML = `<i data-lucide="activity" class="icon-sm"></i> Latencia: ${telemetry.latencia_total_seg}s`;
        lucide.createIcons();
        
    } catch (error) {
        if (qaHistory.contains(skeletonDiv)) {
            qaHistory.removeChild(skeletonDiv);
        }
        addChatMessage(error.message, 'bot', qaHistory);
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
    
    // Renderizar Ficha Técnica (Metadatos Dinámicos) si existen
    if (data.metadatos_adicionales && data.metadatos_adicionales.length > 0) {
        html += `
        <div style="background-color: #f8f9fa; border-left: 4px solid var(--banxico-oro); padding: 10px; margin-bottom: 20px; border-radius: 4px;">
            <h5 style="margin-top: 0; color: #333;">Ficha Técnica / Metadatos Extraídos</h5>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 10px;">
        `;
        data.metadatos_adicionales.forEach(meta => {
            html += `
                <div>
                    <strong style="color: var(--banxico-azul-claro); font-size: 0.9em;">${meta.clave}</strong><br>
                    <span style="font-size: 0.9em;">${meta.valor}</span>
                </div>
            `;
        });
        html += `</div></div>`;
    }

    html += `<div class="table-responsive"><table class="table">
        <thead>
            <tr>
                <th>Campo</th>
                <th>Tipo</th>
                <th>Descripción</th>
                <th>Fórmula / Validaciones</th>
                <th>Catálogo</th>
            </tr>
        </thead>
        <tbody>`;
    
    data.campos_formulario.forEach(campo => {
        let catalogoBadge = campo.es_catalogo ? `<br><span class="badge-catalogo">Catálogo: ${campo.nombre_catalogo_vinculado || 'Sí'}</span>` : '';
        let formulaText = campo.formula_calculo ? `<strong>${campo.formula_calculo}</strong>` : '-';
        let justificacionText = campo.justificacion ? `<br><em style="font-size:0.85em; color:#64748b; display:block; margin-top:4px;"><i data-lucide="info" class="icon-sm"></i> Rationale: ${campo.justificacion}</em>` : '';
        
        html += `<tr>
                    <td><strong>${campo.nombre_campo}</strong>${catalogoBadge}</td>
                    <td>${campo.tipo_dato}</td>
                    <td>${campo.descripcion_funcional}${justificacionText}</td>
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
    
    if (!textosEfimerosGlobal || textosEfimerosGlobal.length === 0) {
        formStatus.innerHTML = "⚠️ Sube un 'Documento Temporal' arriba antes de extraer.";
        formStatus.style.color = "#dc3545"; // rojo
        return;
    }

    formStatus.innerHTML = "Procesando documento... esto puede tomar varios segundos.";
    formStatus.style.color = "var(--banxico-azul-medio)";
    btnSend.disabled = true;
    showTableSkeleton();

    try {
        const response = await fetch('/api/extraer_formulario_full_context', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: "extraer todo", // Dummy
                top_k: 4,
                tema: temaSelect.value,
                textos_efimeros: textosEfimerosGlobal,
                solo_efimero: true, // Siempre true para este flujo
                db_folder: dbSelector ? dbSelector.value : "chroma_db"
            })
        });

        const jsonResp = await response.json();

        if (!response.ok) throw new Error(jsonResp.detail || "Error en el servidor");

        const formResponse = jsonResp.data;
        const telemetry = jsonResp.telemetry;

        renderResultTable(formResponse);
        
        // Agregar info de guardado
        resultContainer.innerHTML += `
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #64748b; text-align: right;">
                <i data-lucide="save" class="icon-sm"></i> Guardado como: <strong>${jsonResp.saved_to}</strong>
            </div>
        `;
        
        formStatus.innerHTML = `¡Listo! Se extrajeron ${formResponse.campos_formulario.length} campos.`;
        formStatus.style.color = "green";

        latencyHud.innerHTML = `<i data-lucide="activity" class="icon-sm"></i> Latencia: ${telemetry.latencia_total_seg || telemetry.latencia_seg}s`;
        lucide.createIcons();
        
    } catch (error) {
        formStatus.innerHTML = error.message;
        formStatus.style.color = "#dc3545";
        resultContainer.innerHTML = `<div class="callout callout-important">${error.message}</div>`;
    } finally {
        btnSend.disabled = false;
    }
});

// Cargar temas dinámicamente desde manifest.yaml
async function loadTemas() {
    try {
        const response = await fetch('/api/temas');
        const result = await response.json();
        if (result.status === 'success' && result.temas) {
            const select = document.getElementById('tema-select');
            result.temas.forEach(tema => {
                const option = document.createElement('option');
                option.value = tema;
                // Formatear: regulacion_general -> Regulacion General
                option.textContent = tema.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                select.appendChild(option);
            });
        }
    } catch (e) {
        console.error("Error cargando temas dinámicos:", e);
    }
}

document.addEventListener("DOMContentLoaded", loadTemas);


// ----------------------------------------------------
// LÓGICA: Pestaña 2 - Extracción Metadatos
// ----------------------------------------------------

metaForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!textosEfimerosGlobal || textosEfimerosGlobal.length === 0) {
        metaStatus.innerHTML = "⚠️ Sube un 'Documento Temporal' arriba antes de extraer.";
        metaStatus.style.color = "#dc3545"; // rojo
        return;
    }

    metaStatus.innerHTML = "Extrayendo metadatos del documento...";
    metaStatus.style.color = "var(--banxico-azul-medio)";
    btnMetaSend.disabled = true;
    
    metaResultContainer.innerHTML = `
        <div class="skeleton skeleton-text" style="width: 80%"></div>
        <div class="skeleton skeleton-text" style="width: 60%; margin-top: 1rem;"></div>
        <div class="skeleton skeleton-text" style="width: 70%; margin-top: 1rem;"></div>
    `;

    try {
        const response = await fetch('/api/extraer_metadatos', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: "extraer metadatos", // Query dummy, el endpoint solo usa textos_efimeros
                textos_efimeros: textosEfimerosGlobal
            })
        });

        const jsonResp = await response.json();

        if (!response.ok) throw new Error(jsonResp.detail || "Error en el servidor");

        const data = jsonResp.data;
        const telemetry = jsonResp.telemetry;
        
        // Renderizar Metadatos
        let subtemasHtml = data.subtemas.map(s => `<span class="badge-catalogo" style="background-color: var(--banxico-azul-medio); margin-right: 4px;">${s}</span>`).join('');
        
        let dinamicosHtml = '';
        if (data.metadatos_dinamicos && data.metadatos_dinamicos.length > 0) {
            dinamicosHtml = `
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px dashed #cbd5e1;">
                    <strong style="color: var(--banxico-oro); font-size: 0.85em; text-transform: uppercase;"><i data-lucide="sparkles" class="icon-sm"></i> Metadatos Dinámicos Detectados</strong>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
            `;
            data.metadatos_dinamicos.forEach(meta => {
                dinamicosHtml += `
                    <div style="background-color: white; padding: 8px; border: 1px solid #e2e8f0; border-radius: 4px;">
                        <strong style="color: #475569; font-size: 0.85em;">${meta.clave}</strong><br>
                        <span style="font-size: 0.9em;">${meta.valor}</span>
                    </div>
                `;
            });
            dinamicosHtml += `</div></div>`;
        }
        
        let html = `
            <div style="background-color: #f8f9fa; border-left: 4px solid var(--banxico-turquesa); padding: 15px; border-radius: 4px;">
                <h4 style="margin-top: 0; color: var(--banxico-azul-institucional); margin-bottom: 1rem;">Metadatos del Documento</h4>
                
                <div style="background-color: #f1f5f9; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 0.9em;">
                    <strong style="color: #64748b;"><i data-lucide="info" class="icon-sm"></i> Justificación de Extracción (Rationale):</strong>
                    <p style="margin: 4px 0 0 0; font-style: italic; color: #475569;">"${data.justificacion_extraccion || 'No proporcionada.'}"</p>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <strong style="color: var(--banxico-azul-claro); font-size: 0.85em; text-transform: uppercase;">Tema Principal</strong>
                        <div style="font-size: 1.1em; font-weight: bold;">${data.tema_principal}</div>
                    </div>
                    <div>
                        <strong style="color: var(--banxico-azul-claro); font-size: 0.85em; text-transform: uppercase;">Confidencialidad</strong>
                        <div style="font-size: 1.1em; font-weight: bold; color: ${data.nivel_confidencialidad.toLowerCase().includes('restringido') ? '#dc3545' : 'inherit'}">${data.nivel_confidencialidad}</div>
                    </div>
                    <div>
                        <strong style="color: var(--banxico-azul-claro); font-size: 0.85em; text-transform: uppercase;">Audiencia Objetivo</strong>
                        <div style="font-size: 1em;">${data.audiencia}</div>
                    </div>
                    <div>
                        <strong style="color: var(--banxico-azul-claro); font-size: 0.85em; text-transform: uppercase;">Frecuencia Reporte</strong>
                        <div style="font-size: 1em;">${data.frecuencia_reporte || 'No especificada'}</div>
                    </div>
                </div>
                
                <div style="margin-top: 15px;">
                    <strong style="color: var(--banxico-azul-claro); font-size: 0.85em; text-transform: uppercase;">Subtemas</strong>
                    <div style="margin-top: 5px;">${subtemasHtml}</div>
                </div>
                
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
                    <strong style="color: var(--banxico-azul-claro); font-size: 0.85em; text-transform: uppercase;">Descripción Corta</strong>
                    <p style="margin-top: 5px; margin-bottom: 0;">${data.descripcion_corta}</p>
                </div>
                ${dinamicosHtml}
            </div>
            
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #64748b; text-align: right;">
                <i data-lucide="save" class="icon-sm"></i> Guardado como: <strong>${jsonResp.saved_to}</strong>
            </div>
        `;
        
        metaResultContainer.innerHTML = html;
        lucide.createIcons();

        metaStatus.innerHTML = `¡Extracción completada con éxito!`;
        metaStatus.style.color = "green";

        latencyHud.innerHTML = `<i data-lucide="activity" class="icon-sm"></i> Latencia: ${telemetry.latencia_seg}s`;
        lucide.createIcons();
        
    } catch (error) {
        metaStatus.innerHTML = error.message;
        metaStatus.style.color = "#dc3545";
        metaResultContainer.innerHTML = `<div class="callout callout-important">${error.message}</div>`;
    } finally {
        btnMetaSend.disabled = false;
    }
});


// ----------------------------------------------------
// (El módulo de evaluaciones se trasladó a Jupyter Notebook)
// ----------------------------------------------------
