import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_frontera_pareto(resultados: list, output_path: str):
    """
    Genera el gráfico de la Frontera de Pareto (Costo vs NDCG).
    
    resultados: lista de dicts con:
      - 'modelo': str
      - 'costo_por_1000': float (Eje X)
      - 'ndcg': float (Eje Y)
    """
    if not resultados:
        print("⚠️ No hay resultados para graficar.")
        return
        
    # Extraer arrays
    nombres = [r["modelo"] for r in resultados]
    costos = [r["costo_por_1000"] for r in resultados]
    ndcgs = [r["ndcg"] for r in resultados]
    
    # Crear la figura
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Graficar puntos
    scatter = sns.scatterplot(x=costos, y=ndcgs, s=150, color="#2ecc71", edgecolor="black", alpha=0.8)
    
    # Etiquetas de los puntos
    for i, txt in enumerate(nombres):
        plt.annotate(txt, (costos[i], ndcgs[i]), 
                     xytext=(10, -5), textcoords='offset points', 
                     fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
                     
    # Encontrar frontera de Pareto
    # Para ser de Pareto, no debe haber otro punto con <= costo Y >= ndcg
    puntos = list(zip(costos, ndcgs, nombres))
    # Ordenar primero por costo (ascendente), luego por ndcg (descendente)
    puntos_ordenados = sorted(puntos, key=lambda x: (x[0], -x[1]))
    
    frontera_costos = []
    frontera_ndcgs = []
    max_ndcg = -1.0
    
    for costo, ndcg, nombre in puntos_ordenados:
        if ndcg > max_ndcg:
            frontera_costos.append(costo)
            frontera_ndcgs.append(ndcg)
            max_ndcg = ndcg
            
    # Trazar la línea roja escalonada de la frontera
    plt.plot(frontera_costos, frontera_ndcgs, color="red", linestyle="--", linewidth=2, label="Frontera de Pareto")
    
    # Configuración estética
    plt.title("Frontera de Pareto: Precisión (NDCG@10) vs Costo Operativo (TCO)", fontsize=14, pad=15)
    plt.xlabel("Costo Operativo Estimado por 1000 consultas (USD)", fontsize=12)
    plt.ylabel("Precisión de Recuperación (NDCG@10)", fontsize=12)
    
    # Forzar límites para que se vea el baseline
    plt.ylim(0, 1.05)
    plt.xlim(left=-0.1) # Para que los modelos gratuitos (Local) se vean despegados del eje Y
    
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Asegurar directorio y guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Grafico de Pareto guardado exitosamente en: {output_path}")
