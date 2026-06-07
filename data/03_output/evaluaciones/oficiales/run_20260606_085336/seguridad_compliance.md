| Riesgo | Control implementado | Evidencia | Riesgo residual |
| :--- | :--- | :--- | :--- |
| Prompt injection/jailbreak | guardrails.py + red-teaming | seguridad_eval.py | Nuevos ataques no cubiertos por set actual |
| PII o consultas sensibles | local-first + no enviar contexto si no es necesario | cascade + tarea QA/formularios | Requiere IAM y redacción de logs |
| Fuga a proveedor | capa factory OpenAI/Ollama y fallback controlado | config_llm.py | Documentar opt-out y contrato institucional |
| Cache incorrecto | juez LLM de equivalencia semántica | generacion.py | Agregar cache de documentos/index versionado |