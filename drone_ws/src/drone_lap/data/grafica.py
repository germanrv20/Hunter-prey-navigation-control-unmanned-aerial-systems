import numpy as np
import matplotlib.pyplot as plt

def generar_grafica_distancia_corregida(archivo_npy):
    """
    Carga datos y genera la gráfica con Distancia en X y Área en Y.
    """
    try:
        # Cargar los datos (.npy)
        datos = np.load(archivo_npy)
        
        # Extraer variables (Asumiendo col 0: Area, col 1: Distancia)
        areas = datos[:, 0]
        distancias = datos[:, 1]
        
        # Crear la figura
        plt.figure(figsize=(8, 5))
        # Intercambiamos el orden en scatter: (x, y) -> (distancia, area)
        plt.scatter(distancias, areas, alpha=0.6, color='tab:blue', label='Datos reales')
        
        # Configuración de ejes con LaTeX
        plt.xlabel(r'Distancia ($m$)', fontsize=12)
        plt.ylabel(r'Área del Bounding Box ($px^2$)', fontsize=12)
        plt.title('Relación Distancia vs Área en Cámara Monocular', fontsize=14)
        
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('grafica_corregida.png', dpi=300)
        print("Gráfica generada: grafica_corregida.png")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generar_grafica_distancia_corregida('datos_area_distancia.npy')
