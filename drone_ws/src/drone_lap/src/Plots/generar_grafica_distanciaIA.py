#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats # Para el R^2

def plot_error_ia():
    # 1. Ruta al archivo
    path = os.path.expanduser('~/drone_ws/src/drone_lap/data/error_ia_dataset.txt')
    
    if not os.path.exists(path):
        print(f"ERROR: No se encuentra el archivo en {path}")
        return

    # 2. Cargar datos
    try:
        data = np.genfromtxt(path, delimiter=' ', invalid_raise=False)
        if data.size == 0: return
        if data.ndim == 1: data = data.reshape(1, -1)
    except Exception as e:
        print(f"Error: {e}")
        return

    real_dist = data[:, 0]
    ia_dist = data[:, 1]
    error_z = data[:, 2]

    # --- CÁLCULOS ESTADÍSTICOS ---
    mae = np.mean(np.abs(error_z))
    rmse = np.sqrt(np.mean(error_z**2))
    slope, intercept, r_value, p_value, std_err = stats.linregress(real_dist, ia_dist)
    r_squared = r_value**2

    # --- CONFIGURACIÓN DE ESTILO ---
    plt.style.use('seaborn-v0_8-whitegrid') # Estilo limpio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Análisis de Rendimiento: Sistema de Visión YOLO', fontsize=16, fontweight='bold')

    # --- GRÁFICA 1: ANÁLISIS DE ERROR (Bland-Altman Lite) ---
    ax1.scatter(real_dist, error_z, alpha=0.6, edgecolors='w', color='#E74C3C', label='Error Instantáneo')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax1.axhline(mae, color='#F39C12', linestyle='--', label=f'MAE ({mae:.2f}m)')
    ax1.axhline(-mae, color='#F39C12', linestyle='--')
    
    # Sombreado de la zona de error medio
    ax1.fill_between(real_dist, -mae, mae, color='#F39C12', alpha=0.1)

    ax1.set_xlabel('Distancia Real Ground Truth (m)', fontsize=11)
    ax1.set_ylabel('Error de Estimación [Real - IA] (m)', fontsize=11)
    ax1.set_title('Distribución del Error vs Distancia', fontsize=13, pad=10)
    ax1.legend(frameon=True, loc='upper right')

    # --- GRÁFICA 2: CORRELACIÓN Y REGRESIÓN ---
    # Puntos de datos
    ax2.scatter(real_dist, ia_dist, alpha=0.6, edgecolors='w', color='#3498DB', label='Estimaciones IA')
    
    # Línea Ideal (y = x)
    lims = [np.min(real_dist), np.max(real_dist)]
    ax2.plot(lims, lims, 'k--', alpha=0.8, linewidth=2, label='Ideal (Referencia)')
    
    # Línea de Tendencia Real
    ax2.plot(real_dist, intercept + slope*real_dist, 'r', alpha=0.5, label='Ajuste Lineal')

    # Cuadro de métricas
    textstr = '\n'.join((
        r'$R^2=%.3f$' % (r_squared, ),
        r'$MAE=%.2f m$' % (mae, ),
        r'$RMSE=%.2f m$' % (rmse, )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax2.set_xlabel('Distancia Real Ground Truth (m)', fontsize=11)
    ax2.set_ylabel('Distancia Estimada por IA (m)', fontsize=11)
    ax2.set_title('Correlación: IA vs Realidad', fontsize=13, pad=10)
    ax2.legend(frameon=True, loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar en alta resolución para el documento
    output_path = os.path.expanduser('~/drone_ws/src/drone_lap/data/grafica_profesional.png')
    plt.savefig(output_path, dpi=300)
    print(f"Gráfica profesional guardada en: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_error_ia()