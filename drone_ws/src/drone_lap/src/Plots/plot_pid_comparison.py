#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURA QUÉ ARCHIVOS QUIERES COMPARAR ---
carpeta_datos = os.path.expanduser("~/drone_ws/src/drone_lap/data/")

pruebas = {
    # YAW
    # "Nombre para la Leyenda": "Nombre del archivo .npy"
    #"PID Óptimo": os.path.join(carpeta_datos, "curvas_YAW_PID_V1_Optimo.npy"),
    #"PID Ondulatorio": os.path.join(carpeta_datos, "curvas_YAW_PID_V1_Lento.npy"),
    #"PID Agresivo": os.path.join(carpeta_datos, "curvas_YAW_PID_V1_agresivo.npy")

    #ALTURA
    # "Nombre para la Leyenda": "Nombre del archivo .npy"
    "PID Óptimo": os.path.join(carpeta_datos, "curvas_ALTURA_PID_V1_Optimo.npy"),
    "PID Lento": os.path.join(carpeta_datos, "curvas_ALTURA_PID_V1_Lento.npy"), 
    "PID Agresivo": os.path.join(carpeta_datos, "curvas_ALTURA_PID_V1_agresivo.npy")

    #DISTANCIA
    #"PID Óptimo": os.path.join(carpeta_datos, "curvas_DISTANCIA_PID_V1_Optimo.npy"),
    #"PID Lento": os.path.join(carpeta_datos, "curvas_DISTANCIA_PID_V1_Lento.npy"), 
    #"PID Agresivo": os.path.join(carpeta_datos, "curvas_DISTANCIA_PID_V1_agresivo.npy")


}

# --- 2. CONFIGURAR LA GRÁFICA ---
plt.figure(figsize=(10, 6))
# Colores cambiados para que el rojo quede reservado para las líneas del 10%
colores = ['blue', 'darkorange', 'green'] 

# Variable para buscar el error máximo y calcular el 10%
pico_maximo = 0.0

# --- 3. DIBUJAR LAS CURVAS ---
for (etiqueta, ruta), color in zip(pruebas.items(), colores):
    if os.path.exists(ruta):
        # Cargamos el archivo
        curvas_test = np.load(ruta, allow_pickle=True)
        
        if len(curvas_test) > 0:
            # Cogemos el PRIMER tirón que hiciste en ese test
            curva_seleccionada = np.array(curvas_test[0]) 
            
            # Separamos en Tiempo y Error
            tiempos = curva_seleccionada[:, 0]
            errores = curva_seleccionada[:, 1]
            
            # Actualizamos el pico máximo para dibujar la banda correctamente
            pico_actual = np.max(np.abs(errores))
            if pico_actual > pico_maximo:
                pico_maximo = pico_actual
            
            # Dibujamos la curva del PID
            plt.plot(tiempos, errores, label=etiqueta, linewidth=2.5, color=color, alpha=0.85)
    else:
        print(f"[Aviso] No se encontró el archivo: {ruta}")

# --- 4. DECORACIÓN (ESTILO PROFESIONAL PARA TFG) ---

# A) Línea del centro (Error Cero)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label="Referencia (Centro)")

# B) Bandas del ±10% (Líneas cortas rojas)
if pico_maximo > 0:
    banda_10 = pico_maximo * 0.05
    # Si prefieres que el 10% sea fijo respecto a la imagen y no al tirón, 
    # borra la línea de arriba y usa esta: banda_10 = 0.10
    
    plt.axhline(y=banda_10, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f"Tolerancia +5%")
    plt.axhline(y=-banda_10, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f"Tolerancia -5%")

# Configurar títulos y etiquetas
plt.title("Comparación de Sintonización de PID (Respuesta al Escalón - YAW)", fontsize=14, fontweight='bold')
plt.xlabel("Tiempo (Segundos)", fontsize=12)
plt.ylabel("Error de Distancia", fontsize=12)
#plt.ylabel("Error de Píxel (Normalizado)", fontsize=12)

# Cuadrícula y Leyenda
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=11, framealpha=0.95)

# Ajustar márgenes y mostrar
plt.tight_layout()
plt.show()