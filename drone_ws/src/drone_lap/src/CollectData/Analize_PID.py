#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def imprimir_estadisticas(nombre, datos):
    """Calcula e imprime estadísticas clave de un array de NumPy"""
    print(f"--- ESTADÍSTICAS: {nombre} ---")
    if len(datos) == 0:
        print("  No hay datos válidos (0 perturbaciones registradas).\n")
        return
    
    media = np.mean(datos)
    mediana = np.median(datos)
    desviacion = np.std(datos)
    minimo = np.min(datos)
    maximo = np.max(datos)
    
    print(f"  Muestras (N)         : {len(datos)}")
    print(f"  Media (Promedio)     : {media:.3f} s")
    print(f"  Mediana              : {mediana:.3f} s")
    print(f"  Desviación Est. (σ)  : {desviacion:.3f} s")
    print(f"  Mínimo (Más rápido)  : {minimo:.3f} s")
    print(f"  Máximo (Más lento)   : {maximo:.3f} s\n")

def main(archivo_npy):
    if not os.path.exists(archivo_npy):
        print(f"Error: No se encontró el archivo '{archivo_npy}'")
        return

    print("==================================================")
    print("  ANÁLISIS DE TIEMPOS DE ESTABLECIMIENTO (PID)")
    print(f"  Leyendo: {archivo_npy}")
    print("==================================================\n")

    # 1. Cargar la matriz
    try:
        matriz = np.load(archivo_npy)
    except Exception as e:
        print(f"Error al cargar el archivo .npy: {e}")
        return

    # 2. Extraer columnas (0: Yaw, 1: Altura, 2: Distancia)
    ts_x_raw = matriz[:, 0]
    ts_y_raw = matriz[:, 1]
    ts_z_raw = matriz[:, 2]

    # 3. Limpiar los valores NaN (huecos vacíos que añadimos para cuadrar la matriz)
    ts_x = ts_x_raw[~np.isnan(ts_x_raw)]
    ts_y = ts_y_raw[~np.isnan(ts_y_raw)]
    ts_z = ts_z_raw[~np.isnan(ts_z_raw)]

    # 4. Imprimir Análisis Estadístico
    imprimir_estadisticas("YAW (X) - Control Horizontal", ts_x)
    imprimir_estadisticas("ALTURA (Y) - Control Vertical", ts_y)
    imprimir_estadisticas("DISTANCIA (Z) - Control de Avance", ts_z)

    # 5. Generar la Gráfica (Boxplot)
    datos_grafica = [ts_x, ts_y, ts_z]
    etiquetas = ['Yaw (X)', 'Altura (Y)', 'Distancia (Z)']

    # Crear figura
    plt.figure(figsize=(9, 6))
    
    # Dibujar boxplot
    box = plt.boxplot(datos_grafica, labels=etiquetas, patch_artist=True, 
                      boxprops=dict(facecolor='lightblue', color='navy'),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(color='navy', linewidth=1.5),
                      capprops=dict(color='navy', linewidth=1.5),
                      flierprops=dict(marker='o', color='red', alpha=0.5))
                      
    # Añadir colores distintos a cada caja para que quede más visual
    colores = ['#ff9999', '#66b3ff', '#99ff99']
    for parche, color in zip(box['boxes'], colores):
        parche.set_facecolor(color)

    plt.title('Distribución de Tiempos de Establecimiento del Dron', fontsize=14, fontweight='bold')
    plt.ylabel('Tiempo de Establecimiento (segundos)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir un texto explicando cómo leer la gráfica
    texto_explicativo = (
        "La línea roja indica la Mediana.\n"
        "La caja representa el 50% central de los datos (Q1 a Q3).\n"
        "Los puntos aislados son valores atípicos (outliers)."
    )
    plt.figtext(0.15, 0.02, texto_explicativo, fontsize=9, color='gray', style='italic')
    plt.subplots_adjust(bottom=0.15) # Dejar espacio para el texto inferior

    # 6. Guardar la gráfica automáticamente como PNG en la misma carpeta
    ruta_grafica = archivo_npy.replace('.npy', '_boxplot.png')
    plt.savefig(ruta_grafica, dpi=300, bbox_inches='tight')
    print(f"==================================================")
    print(f"¡Gráfica generada y guardada en:\n{ruta_grafica}")
    print("==================================================")
    
    # Mostrar la gráfica por pantalla
    plt.show()

if __name__ == '__main__':
    # Si no le pasamos un archivo por terminal, por defecto usa la ruta que me has indicado
    ruta_por_defecto = "/home/germanrv/drone_ws/src/drone_lap/data/matriz_tiempos_vuelo_vivo.npy"
    
    if len(sys.argv) > 1:
        ruta_archivo = sys.argv[1]
    else:
        ruta_archivo = ruta_por_defecto
        
    main(ruta_archivo)
