import os
import numpy as np
import glob

def main():
    # 1. Definir la carpeta donde están tus datos
    data_dir = os.path.expanduser("~/drone_ws/src/drone_lap/data/")
    
    # 2. Buscar todos los archivos que terminen en .npy
    # (Asegúrate de no incluir el archivo final combinado si ya existe)
    archivos = glob.glob(os.path.join(data_dir, "distance_*.npy")) # Ajusta el patrón si es necesario
    
    if not archivos:
        print("No se encontraron archivos .npy para unir.")
        return

    print(f"Se han encontrado {len(archivos)} archivos para fusionar.")

    lista_arrays = []
    
    # 3. Cargar cada archivo y meterlo en una lista
    for archivo in archivos:
        try:
            data = np.load(archivo)
            # Verificamos que tenga la forma correcta (N filas, 2 columnas)
            if len(data.shape) == 2 and data.shape[1] == 2:
                lista_arrays.append(data)
                print(f"Cargado: {os.path.basename(archivo)} -> Filas: {data.shape[0]}")
            else:
                print(f"IGNORADO: {os.path.basename(archivo)} no tiene 2 columnas.")
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")

    if not lista_arrays:
        print("No hay datos válidos para fusionar.")
        return

    # 4. LA MAGIA: Unir todos los arrays uno debajo del otro (Sin sobreponer)
    dataset_final = np.vstack(lista_arrays)

    # 5. Guardar el nuevo súper-archivo
    archivo_salida = os.path.join(data_dir, "super_dataset_distancia.npy")
    np.save(archivo_salida, dataset_final)

    print("-" * 50)
    print("¡Fusión completada con éxito!")
    print(f"Total de datos (Filas): {dataset_final.shape[0]}")
    print(f"Archivo guardado en: {archivo_salida}")
    print("-" * 50)

if __name__ == "__main__":
    main()