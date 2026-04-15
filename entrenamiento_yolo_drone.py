# -*- coding: utf-8 -*-
"""
Script de Entrenamiento YOLO - Adaptado para entorno local
"""

import os
import glob
import torch
import yaml
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
from roboflow import Roboflow

# ==========================================
# 0. FUNCIONES AUXILIARES
# ==========================================

def redistribuir_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    print(f"\n[INFO] INICIANDO REDISTRIBUCIÓN desde: {input_dir}")

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_pairs = []
    splits = ['train', 'valid', 'test']

    for split in splits:
        img_dir = os.path.join(input_dir, split, 'images')
        lbl_dir = os.path.join(input_dir, split, 'labels')

        if not os.path.exists(img_dir):
            continue

        files = os.listdir(img_dir)
        count_split = 0

        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_extensions:
                img_path = os.path.join(img_dir, f)
                lbl_path = os.path.join(lbl_dir, name + ".txt")

                if os.path.exists(lbl_path):
                    all_pairs.append((img_path, lbl_path))
                    count_split += 1

        print(f"   Recolectadas {count_split} parejas de '{split}'")

    total_files = len(all_pairs)
    print(f"[INFO] TOTAL DE PARES ENCONTRADOS: {total_files}")

    if total_files == 0:
        print("[ERROR] No se encontraron pares imagen/etiqueta. Verifica las rutas.")
        return None

    random.shuffle(all_pairs)

    i_train = int(total_files * train_ratio)
    i_val = i_train + int(total_files * val_ratio)

    datasets = {
        'train': all_pairs[:i_train],
        'valid': all_pairs[i_train:i_val],
        'test': all_pairs[i_val:]
    }

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print(f"\n[INFO] COPIANDO ARCHIVOS A: {output_dir} ...")

    for split, pairs in datasets.items():
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

        for img_src, lbl_src in pairs:
            shutil.copy(img_src, os.path.join(output_dir, split, 'images', os.path.basename(img_src)))
            shutil.copy(lbl_src, os.path.join(output_dir, split, 'labels', os.path.basename(lbl_src)))

        print(f"   {split.upper()}: {len(pairs)} imágenes generadas.")

    nombres_clases = []
    yaml_orig = os.path.join(input_dir, 'data.yaml')
    if os.path.exists(yaml_orig):
        try:
            with open(yaml_orig, 'r') as f:
                d = yaml.safe_load(f)
                nombres_clases = d.get('names', [])
        except: pass

    new_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': nombres_clases,
        'nc': len(nombres_clases)
    }

    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(new_yaml, f)

    print(f"\n[INFO] PROCESO COMPLETADO. Nuevo dataset está en: {output_dir}")
    return output_dir

def analizar_yolo_dataset(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    class_names = []

    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            names = data.get('names', [])
            if isinstance(names, dict):
                class_names = [names[i] for i in sorted(names.keys())]
            else:
                class_names = names
        print(f"[INFO] Clases detectadas: {class_names}")
    else:
        print("[ADVERTENCIA] No se encontró 'data.yaml'. Se usarán IDs numéricos.")

    print("\n[INFO] Generando estadísticas...")
    label_files = glob.glob(f'{dataset_dir}/**/labels/*.txt', recursive=True)
    class_counts = Counter()

    for lfile in label_files:
        with open(lfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        if class_names and 0 <= cls_id < len(class_names):
                            label = class_names[cls_id]
                        else:
                            label = str(cls_id)
                        class_counts[label] += 1
                    except ValueError: pass

    if class_counts:
        print(f"\n{'CLASE':<30} | {'CANTIDAD':<10}")
        print("-" * 45)
        sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls, count in sorted_items:
            print(f"{cls:<30} | {count:<10}")

        keys = [k for k, v in sorted_items]
        vals = [v for k, v in sorted_items]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=keys, y=vals, hue=keys, legend=False, palette="viridis")
        plt.title(f"Distribución de Etiquetas (Total: {sum(vals)})")
        plt.xlabel("Clase")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    print("\n[INFO] Generando ejemplos visuales...")
    all_images = glob.glob(f'{dataset_dir}/**/images/*.jpg', recursive=True) + \
                 glob.glob(f'{dataset_dir}/**/images/*.png', recursive=True) + \
                 glob.glob(f'{dataset_dir}/**/images/*.jpeg', recursive=True)

    if all_images:
        samples = random.sample(all_images, min(len(all_images), 6))
        plt.figure(figsize=(15, 10))

        for i, img_path in enumerate(samples):
            img = cv2.imread(img_path)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            label_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + ".txt"
            if not os.path.exists(label_path):
                 label_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', os.path.basename(img_path).rsplit('.', 1)[0] + ".txt")

            box_drawn = False
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                cx, cy, bw, bh = map(float, parts[1:5])

                                x1 = int((cx - bw/2) * w)
                                y1 = int((cy - bh/2) * h)
                                x2 = int((cx + bw/2) * w)
                                y2 = int((cy + bh/2) * h)

                                color = plt.cm.tab10(cls_id % 10)
                                color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

                                cv2.rectangle(img, (x1, y1), (x2, y2), color_rgb, 3)

                                if class_names and cls_id < len(class_names):
                                    label_txt = class_names[cls_id]
                                else:
                                    label_txt = str(cls_id)

                                text_y = y1 - 10 if y1 > 25 else y1 + 25
                                cv2.putText(img, label_txt, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
                                cv2.putText(img, label_txt, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                                box_drawn = True
                            except ValueError: pass

            plt.subplot(2, 3, i+1)
            plt.imshow(img)
            plt.axis('off')

            parent_folder = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            if box_drawn:
                plt.title(f"[{parent_folder}] Detectado", fontsize=10)
            else:
                plt.title(f"[{parent_folder}] Sana/Fondo", color='green', fontsize=10)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2) # Pausa breve para mostrar y continuar

def analizar_estructura(dataset_dir):
    if not dataset_dir or not os.path.exists(dataset_dir):
        print("[ERROR] El directorio del dataset no es válido.")
        return

    print(f"\nREPORTE DEL DATASET (Carpeta: {dataset_dir})")
    print("-" * 50)
    carpetas = ['train', 'valid', 'test']
    total_imgs = 0

    for carpeta in carpetas:
        ruta_split = os.path.join(dataset_dir, carpeta)
        ruta_imgs = os.path.join(ruta_split, 'images')

        if not os.path.exists(ruta_imgs):
            ruta_imgs = ruta_split

        if os.path.exists(ruta_imgs) and os.path.isdir(ruta_imgs):
            imgs = glob.glob(os.path.join(ruta_imgs, '*.jpg')) + \
                   glob.glob(os.path.join(ruta_imgs, '*.jpeg')) + \
                   glob.glob(os.path.join(ruta_imgs, '*.png'))
            n = len(imgs)
            total_imgs += n
            print(f"{carpeta.upper()}: {n} imágenes encontradas.")
        else:
            if carpeta != 'test':
                print(f"{carpeta.upper()}: No encontrada.")

    print("-" * 50)
    print(f"TOTAL: {total_imgs} imágenes.")

def configurar_yaml(dataset_dir):
    if not dataset_dir: return
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            data['path'] = os.path.abspath(dataset_dir)
            data['train'] = "train/images"
            data['val'] = "valid/images"
            data['test'] = "test/images"

            with open(yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            print(f"\n[INFO] Archivo corregido: {yaml_path}")
        except Exception as e:
            print(f"[ERROR] Editando yaml: {e}")
    else:
        print("[ERROR] No se encontró 'data.yaml'.")


# ==========================================
# FLUJO PRINCIPAL DE EJECUCIÓN
# ==========================================
if __name__ == '__main__':
    
    # 1. Descarga del Dataset
    print("\n[INFO] Conectando con Roboflow...")
    rf = Roboflow(api_key="RApFtl6W6NZRZvedDpz2")
    project = rf.workspace("germanrv-uyz9j").project("synthetic-drone-d2ius")
    version = project.version(6)
    dataset = version.download("yolov8")

    dataset_dir = dataset.location
    print(f"\n[INFO] Descarga completada en: {dataset_dir}")

    if dataset_dir:
       configurar_yaml(dataset_dir)

    print(f"\nContenido de {dataset_dir}:")
    print(os.listdir(dataset_dir))
    
    dataset_dir_final = redistribuir_dataset(dataset_dir, "dataset_repartido")
    analizar_yolo_dataset(dataset_dir_final)
    analizar_estructura(dataset_dir_final)

    # 2. Torneo de Modelos (Evaluación)
    if not dataset_dir_final or not os.path.exists(dataset_dir_final):
        print("[ERROR] La variable 'dataset_dir_final' no está definida o no existe.")
    else:
        print(f"\n[INFO] Usando dataset en: {dataset_dir_final}")
        ruta_yaml = os.path.join(dataset_dir_final, "data.yaml")

        # Nota: yolo26n.pt no existe oficialmente en Ultralytics, es posible que arroje error.
        modelos_a_entrenar = [
            'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
            'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt',
            'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt',
            'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt'
        ]

        PROYECTO = 'runs/detect'
        resultados_metricas = []

        for nombre_pesos in modelos_a_entrenar:
            nombre_modelo = nombre_pesos.split('.')[0]
            print(f"\n{'='*60}")
            print(f" INICIANDO ENTRENAMIENTO CON: {nombre_modelo.upper()}")
            print(f"{'='*60}\n")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] Dispositivo detectado: {device}")

            try:
                model = YOLO(nombre_pesos)
                model.train(
                    data=ruta_yaml,
                    epochs=25,
                    imgsz=512,
                    batch=16,
                    project=PROYECTO,
                    name=f"entrenamiento_{nombre_modelo}",
                    plots=False,
                    device=device
                )

                print(f"\n Extrayendo métricas de {nombre_modelo}...")
                metrics = model.val()

                tiempo_inferencia = metrics.speed['inference']
                map50 = metrics.box.map50
                map50_95 = metrics.box.map

                resultados_metricas.append({
                    'Modelo': nombre_modelo,
                    'mAP50': map50,
                    'mAP50-95': map50_95,
                    'Inferencia_ms': tiempo_inferencia
                })

            except Exception as e:
                print(f" Error al evaluar {nombre_modelo}: {e}")
                print("Saltando al siguiente modelo...")

        print("\n[INFO] ENTRENAMIENTO DE TODOS LOS MODELOS FINALIZADO.")

        if len(resultados_metricas) > 0:
            df_metricas = pd.DataFrame(resultados_metricas)
            print("\n TABLA DE RESULTADOS:")
            print(df_metricas.to_string(index=False))

            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(14, 8))
            ax = sns.scatterplot(data=df_metricas, x='Inferencia_ms', y='mAP50', hue='Modelo', s=300, palette="tab20")
            plt.title('Decisión para Dron: Precisión (mAP50) vs Velocidad de Inferencia', fontsize=16, fontweight='bold')
            plt.xlabel('Tiempo de Inferencia por imagen (ms)', fontsize=12)
            plt.ylabel('Precisión (mAP50)', fontsize=12)

            for i in range(df_metricas.shape[0]):
                plt.text(df_metricas['Inferencia_ms'][i] + 0.1,
                         df_metricas['mAP50'][i] + 0.005,
                         df_metricas['Modelo'][i],
                         fontsize=10)

            plt.axvline(x=df_metricas['Inferencia_ms'].median(), color='red', linestyle='--', alpha=0.5)
            plt.axhline(y=df_metricas['mAP50'].median(), color='red', linestyle='--', alpha=0.5)
            plt.text(df_metricas['Inferencia_ms'].min(), df_metricas['mAP50'].max(),
                     'Zona Ideal (Rápido y Preciso)', color='green', fontsize=12, alpha=0.8, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(12, 6))
            df_sorted = df_metricas.sort_values('mAP50', ascending=False)
            sns.barplot(data=df_sorted, x='mAP50', y='Modelo', palette='viridis')
            plt.title('Ranking de Precisión (mAP50) por Modelo', fontsize=16, fontweight='bold')
            plt.xlabel('mAP50 (0 a 1)')
            plt.ylabel('Modelo')
            plt.tight_layout()
            plt.show()

    # 3. Entrenamiento Final y Prueba Local
    print("\n[INFO] Iniciando entrenamiento final de prueba con YOLOv8n...")
    model = YOLO('yolov8n.pt')
    model.train(
        data=ruta_yaml,
        epochs=20, # Cambia este valor para el entrenamiento real
        imgsz=512,
        batch=16,
        project=PROYECTO,
        name='entrenamiento_drone_final',
        plots=True,
        device=device
    )
    print("Entrenamiento finalizado.")

    # Inferencia de prueba interactiva
    print("\n" + "="*50)
    ruta_prueba = input("⬇ Introduce la ruta de una imagen local para probar el modelo (o presiona Enter para salir): ")
    
    if ruta_prueba and os.path.exists(ruta_prueba):
        print(f"\nAnalizando imagen: {ruta_prueba} ...")
        # Usamos el modelo recién entrenado
        results = model.predict(source=ruta_prueba, conf=0.1, save=True)

        for result in results:
            imagen_con_cajas = result.plot()
            
            # Mostrar usando OpenCV de escritorio
            cv2.imshow("Resultado YOLO", imagen_con_cajas)
            print("\n[!] Presiona cualquier tecla en la ventana de la imagen para cerrarla.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if len(result.boxes) == 0:
                print(" No se detectó nada. Intenta bajar el 'conf' o usar una foto más clara.")
            else:
                for box in result.boxes:
                    clase = result.names[int(box.cls[0])]
                    confianza = float(box.conf[0])
                    print(f"--> Detectado: {clase} ({confianza:.1%})")
    else:
        print("\nPrueba de inferencia omitida o ruta no válida. ¡Proceso completado!")
