#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def main(archivo_npy):
    if not os.path.exists(archivo_npy):
        print(f"Error: No se encontró el archivo '{archivo_npy}'")
        return

    print("==================================================")
    print("  ANÁLISIS DE RENDIMIENTO DE IA (DISTANCIA)")
    print(f"  Leyendo: {archivo_npy}")
    print("==================================================\n")

    # 1. Cargar la matriz
    try:
        matriz = np.load(archivo_npy)
    except Exception as e:
        print(f"Error al cargar el archivo .npy: {e}")
        return

    # Validar que tiene al menos 3 columnas (El logger guarda 4)
    if matriz.shape[1] < 3:
        print("Error: El archivo no tiene el formato esperado [Tiempo, IA, Real, Error_Z]")
        return

    # 2. Extraer columnas (1: Distancia IA, 2: Distancia Real)
    dist_ia_raw = matriz[:, 1]
    dist_real_raw = matriz[:, 2]

    # 3. Filtrar datos anómalos (YOLO perdido manda -999.0) o NaN
    indices_validos = np.where((dist_ia_raw > 0) & (~np.isnan(dist_ia_raw)))[0]
    
    if len(indices_validos) == 0:
        print("No hay datos de inferencia válidos en este archivo.")
        return

    dist_ia = dist_ia_raw[indices_validos]
    dist_real = dist_real_raw[indices_validos]

    # 4. Calcular el Error de la IA (Error Absoluto)
    error_ia = np.abs(dist_ia - dist_real)

    # 5. Estadísticas por pantalla
    print(f"--- ESTADÍSTICAS GLOBALES DE LA RED NEURONAL ---")
    print(f"  Total de inferencias válidas : {len(error_ia)} frames")
    print(f"  Error Medio Absoluto (MAE)   : {np.mean(error_ia):.4f} metros")
    print(f"  Error Máximo Cometido        : {np.max(error_ia):.4f} metros")
    
    # 6. Agrupar por tramos para la terminal
    print(f"\n--- ERROR MEDIO POR TRAMOS DE DISTANCIA ---")
    tramos = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    for i in range(len(tramos)-1):
        inf, sup = tramos[i], tramos[i+1]
        idx_tramo = np.where((dist_real >= inf) & (dist_real < sup))[0]
        if len(idx_tramo) > 0:
            mae_tramo = np.mean(error_ia[idx_tramo])
            print(f"  [{inf:.1f}m - {sup:.1f}m]: MAE = {mae_tramo:.4f} m (Muestras: {len(idx_tramo)})")
        else:
            print(f"  [{inf:.1f}m - {sup:.1f}m]: Sin datos.")

    # 7. Generar la Gráfica
    plt.figure(figsize=(10, 6))
    
    # Nube de puntos (con transparencia alpha=0.3 para ver densidad)
    plt.scatter(dist_real, error_ia, alpha=0.3, color='dodgerblue', edgecolor='none', label='Muestras de vuelo (IA vs Real)')

    # Línea de tendencia (Media Móvil)
    # Usamos pandas para ordenar por distancia real y sacar una línea suavizada
    df_plot = pd.DataFrame({'Real': dist_real, 'Error': error_ia}).sort_values(by='Real')
    # Ajusta el window si quieres la curva más o menos suavizada
    df_plot['Tendencia'] = df_plot['Error'].rolling(window=100, min_periods=1, center=True).mean()
    
    plt.plot(df_plot['Real'], df_plot['Tendencia'], color='red', linewidth=3, label='Tendencia (Error Medio)')

    # Estilos del gráfico
    plt.title('Rendimiento IA: Error de Estimación vs Distancia Real', fontsize=14, fontweight='bold')
    plt.xlabel('Distancia Real entre UAVs (metros)', fontsize=12)
    plt.ylabel('Error Absoluto de la IA (metros)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11, loc='upper left')
    
    # Restringir eje Y si hay algún pico loco de error que rompa la escala (opcional)
    # plt.ylim(0, 1.5)

    # 8. Guardar la gráfica
    ruta_grafica = archivo_npy.replace('.npy', '_grafica_IA.png')
    plt.savefig(ruta_grafica, dpi=300, bbox_inches='tight')
    print(f"\n==================================================")
    print(f"¡Gráfica generada y guardada en:\n{ruta_grafica}")
    print("==================================================")
    
    plt.show()

if __name__ == '__main__':
    # Ruta por defecto al archivo que acabas de generar
    ruta_por_defecto = "/home/germanrv/drone_ws/src/drone_lap/data/vuelo_20260527_131825.npy"
    
    if len(sys.argv) > 1:
        ruta_archivo = sys.argv[1]
    else:
        ruta_archivo = ruta_por_defecto
        
    main(ruta_archivo)
