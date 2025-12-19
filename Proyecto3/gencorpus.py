import pandas as pd
import os
import sys

def limpiar_y_estructurar(row):
    """
    Convierte la fila en texto narrativo.
    Usa .get() para evitar errores si falta algún dato opcional.
    """
    # Protección contra datos vacíos
    id_registro = row.get('id', 'SinID')
    usuario = row.get('usuario', 'Anónimo')
    fecha = row.get('fecha', 'SinFecha')
    likes = row.get('likes', 0)
    reposts = row.get('reposts', 0)
    tema = row.get('tema', 'General')
    texto = row.get('texto', '')
    emocion = row.get('sentimiento', 'neutral')

    # Determinamos el tono emocional
    if emocion == 'negativo':
        contexto_emocional = "El usuario expresa frustración o malestar."
    elif emocion == 'positivo':
        contexto_emocional = "El usuario muestra optimismo o acuerdo."
    else:
        contexto_emocional = "El usuario mantiene un tono neutral/reflexivo."

    # Construimos el bloque
    return f"""
=== REGISTRO ID: {id_registro} ===
[METADATOS]
- Autor: {usuario}
- Fecha: {fecha}
- Impacto Social: {likes} Likes, {reposts} Reposts
- Tema Clave: {tema}
- Análisis de Sentimiento: {str(emocion).upper()} ({contexto_emocional})

[TESTIMONIO]
"{texto}"
=================================
"""

def generar_corpus():
    print("--- INICIANDO GENERACIÓN DE CORPUS ---")
    archivo_csv = 'dataset_sintetico_5000_ampliado.csv'
    archivo_salida = 'CORPUS_FILOSOFICO_GEN_Z.txt'
    
    try:
        print(f"1. Leyendo {archivo_csv}...")
        
        # --- CORRECCIÓN 1: encoding='utf-8-sig' maneja el BOM de Excel ---
        try:
            df = pd.read_csv(archivo_csv, encoding='utf-8-sig')
        except:
            # Si falla, intenta con encoding estándar
            df = pd.read_csv(archivo_csv, encoding='latin1')

        # --- CORRECCIÓN 2: LIMPIEZA DE COLUMNAS (La solución al error) ---
        # Quitamos espacios en blanco y convertimos a minúsculas
        # Así ' id ' o 'ID' se convierten en 'id'
        df.columns = df.columns.str.strip().str.lower()
        
        print(f"   Columnas detectadas: {list(df.columns)}") # Para debug
        
        # Verificamos que 'id' exista, si no, creamos un índice falso
        if 'id' not in df.columns:
            print("   AVISO: No se encontró columna 'id', generando una automática...")
            df['id'] = range(1, len(df) + 1)

        # Limpieza de duplicados
        cantidad_inicial = len(df)
        df = df.drop_duplicates(subset=['texto'])
        print(f"   -> Se eliminaron {cantidad_inicial - len(df)} duplicados.")
        
        # Generación
        print("2. Estructurando la información...")
        lista_corpus = df.apply(limpiar_y_estructurar, axis=1)
        
        # Guardado
        print(f"3. Guardando en {archivo_salida}...")
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write("CORPUS DE ANÁLISIS: FILOSOFÍA Y GENERACIÓN Z\n")
            f.write(f"Fuente: {archivo_csv}\n")
            f.write("Marcos: Byung-Chul Han, Foucault, Bauman\n")
            f.write("\n" + "#"*50 + "\n\n")
            
            for item in lista_corpus:
                f.write(item)
                f.write("\n")
                
        print("\n¡ÉXITO TOTAL! Corpus generado.")
        print(f"--> Revisa el archivo: {archivo_salida}")
        
    except Exception as e:
        print(f"\nERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generar_corpus()