import numpy as np
import matplotlib.pyplot as plt
import os

def generar_comparativa():
    dir_script = os.path.dirname(os.path.abspath(__file__))
    ruta_data = os.path.join(dir_script, "..", "data", "distance_data_v2.npy")

    if not os.path.exists(ruta_data):
        print(f"Error: No existe el archivo en {ruta_data}")
        return

    try:
        datos = np.load(ruta_data)
        
        # Columna 0: Area, Columna 1: Distancia
        areas = datos[:, 0]
        distancias = datos[:, 1]

        # Aplicar el "truco" de linealización (1 / raiz(Area))
        # Filtramos para evitar ceros
        mask = areas > 0
        areas_f = areas[mask]
        dist_f = distancias[mask]
        truco = 1.0 / np.sqrt(areas_f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Grafica 1: Sin el truco ---
        ax1.scatter(distancias, areas, alpha=0.5, color='tab:blue')
        ax1.set_title('Relacion Cruda: Area vs Distancia')
        ax1.set_xlabel('Distancia (m)')
        ax1.set_ylabel('Area (px^2)')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- Grafica 2: Con el truco ---
        ax2.scatter(dist_f, truco, alpha=0.5, color='tab:red')
        # Cambiamos \text{Area} por Area para evitar el error de parsing
        ax2.set_title(r'Relacion Linealizada: $1/\sqrt{Area}$ vs Distancia')
        ax2.set_xlabel('Distancia (m)')
        ax2.set_ylabel(r'$1/\sqrt{Area}$')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig("comparacion_linealizacion.png", dpi=300)
        print("Grafica comparativa generada con exito como 'comparacion_linealizacion.png'")

    except Exception as e:
        print(f"Error al procesar los datos: {e}")

if __name__ == "__main__":
    generar_comparativa()
