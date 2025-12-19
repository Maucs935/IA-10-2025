import os
import cv2
import numpy as np
import shutil

# --- CONFIGURACIÓN ACTUALIZADA ---
# Usamos r'' para que Python lea bien las rutas de Windows
DIR_ORIGEN = r'C:\Users\oem\Downloads\IANOT\dataset'
DIR_DESTINO = r'C:\Users\oem\Downloads\IANOT\datasetreco'
IMG_SIZE = 224

def resize_with_padding(image, target_size):
    """Redimensiona con padding negro para no deformar la imagen"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Redimensionar manteniendo proporción
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Crear lienzo negro
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calcular posición centrada
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # Pegar imagen
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
    return canvas

print(f"🛠️  Iniciando procesamiento...")
print(f"📂 Origen: {DIR_ORIGEN}")
print(f"💾 Destino: {DIR_DESTINO}")

# Crear carpeta principal de destino si no existe
if not os.path.exists(DIR_DESTINO):
    os.makedirs(DIR_DESTINO)

total_procesadas = 0
errores = 0

# Recorrer carpetas dentro de 'Recorte'
# OJO: Este script espera que dentro de 'Recorte' haya subcarpetas (ej: Recorte/Gatos/foto.jpg)
contenido = os.listdir(DIR_ORIGEN)

# Si la carpeta está vacía o no tiene subcarpetas, avisamos
subcarpetas = [f for f in contenido if os.path.isdir(os.path.join(DIR_ORIGEN, f))]
if not subcarpetas:
    print("\n⚠️  ADVERTENCIA: No se encontraron subcarpetas dentro de 'Recorte'.")
    print("   Este script está diseñado para datasets organizados (ej: Recorte/Clase1/foto.jpg).")
    print("   Si tus imágenes están sueltas directamente en 'Recorte', avísame para ajustar el código.\n")

for folder_name in subcarpetas:
    path_origen = os.path.join(DIR_ORIGEN, folder_name)
    path_destino = os.path.join(DIR_DESTINO, folder_name)
    
    # Crear la carpeta correspondiente en el destino
    if not os.path.exists(path_destino):
        os.makedirs(path_destino)
        
    print(f"--> Procesando categoría: {folder_name}...")
    
    # Procesar archivos
    files = os.listdir(path_origen)
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                # 1. Leer imagen original
                img_path = os.path.join(path_origen, filename)
                # Usamos imdecode para evitar problemas con tildes/ñ en rutas de Windows
                stream = open(img_path, "rb")
                bytes_img = bytearray(stream.read())
                numpyarray = np.asarray(bytes_img, dtype=np.uint8)
                img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    print(f"   [!] Imagen corrupta o formato no soportado: {filename}")
                    errores += 1
                    continue

                # Si la imagen tiene canal alpha (PNG transparente), quitarlo para evitar errores al pegar en negro
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # 2. Procesar (Resize + Padding)
                img_processed = resize_with_padding(img, IMG_SIZE)
                
                # 3. Guardar en la nueva carpeta
                cv2.imwrite(os.path.join(path_destino, filename), img_processed)
                total_procesadas += 1
                
            except Exception as e:
                print(f"   [X] Error en {filename}: {e}")
                errores += 1

print("\n" + "="*40)
print(f"✅ PROCESO TERMINADO")
print(f"📊 Imágenes guardadas correctamente: {total_procesadas}")
print(f"⚠️  Errores/Archivos corruptos: {errores}")
print(f"📁 Tu nuevo dataset está en: {DIR_DESTINO}")
print("="*40)