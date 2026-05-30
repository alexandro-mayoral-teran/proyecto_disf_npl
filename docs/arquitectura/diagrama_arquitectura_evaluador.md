# ⚖️ Arquitectura del Módulo Evaluador (LLM-as-a-Judge)

Este diagrama detalla cómo opera el orquestador de pruebas (`evaluador.py`), simulando un laboratorio de MLOps automatizado que corre cientos de pruebas, mide costos y grafica la Frontera de Pareto.

---

## 1. Topología del Pipeline de Evaluación (Mermaid)

```mermaid
graph TD
    %% Base de la Verdad
    A[("Golden Dataset<br>110 Consultas Humanas")] --> B

    %% Orquestador de Evaluación
    B["evaluador.py<br>Orquestador de Pruebas"] -->|1. Simula Usuario| C["retrieval.py<br>(Motor de Búsqueda)"]
    C -.->|Recupera| D[("ChromaDB<br>Vector Store")]
    
    %% Juez
    B -->|2. Inyecta Contexto| E["LLM-as-a-Judge<br>(GPT-4o)"]
    E -->|Analiza y Falla| F{"Taxonomía de Error"}
    
    %% Ramas de Taxonomía
    F -->|Texto no Recuperado| G["A - Fallo de Recuperación"]
    F -->|LLM Alucinó/Ignoró| H["B - Fallo de Generación"]
    F -->|JSON Roto| I["C - Fallo Estructural"]

    %% Métricas y Salida
    B -->|3. Manda Datos| J["telemetria.py<br>(Costos y Latencia)"]
    B -->|4. Aplica Algoritmo| K["Bootstrap CI (95%)<br>Significancia Estadística"]
    J --> L
    K --> L["graficos.py<br>Generador Visual"]
    L --> M[("Frontera de Pareto<br>Gráfica Final")]

    %% Estilos
    style B fill:#e74c3c,stroke:#c0392b,stroke-width:4px,color:white
    style A fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:black
    style E fill:#34495e,stroke:#2c3e50,stroke-width:2px,color:white
    style M fill:#2ecc71,stroke:#27ae60,stroke-width:4px,color:black
```

---

## 2. Flujo Explicado

1. **Ingesta de la Verdad:** El evaluador lee el *Ground Truth* (las 110 preguntas curadas por expertos con sus respuestas exactas esperadas).
2. **Simulación de Tráfico:** `evaluador.py` agarra cada pregunta y se la lanza al sistema real de búsqueda (`retrieval.py`). Prende un cronómetro para medir la latencia y cuenta los tokens gastados enviando esos datos a `telemetria.py`.
3. **El Juez Implacable:** Los fragmentos recuperados y la respuesta generada se envían a un Juez (típicamente el modelo más inteligente, como GPT-4o). Este juez no usa Regex, usa razonamiento para emitir un veredicto en 3 niveles (Taxonomía A/B/C).
4. **Validación Científica:** Los puntajes individuales de NDCG se pasan por un algoritmo de **Bootstrapping** para calcular intervalos de confianza al 95%.
5. **Decisión Final:** Todos los datos (Costos de telemetría vs NDCG con Bootstrapping) convergen en `graficos.py`, el cual dibuja la Frontera de Pareto para poder elegir la arquitectura ganadora objetivamente.
