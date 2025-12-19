import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import sys

# --- CONFIGURACIÓN ---
client = OpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="lm-studio"
)

def cargar_datos():
    print("1. Cargando y procesando el dataset...")
    try:
        df = pd.read_csv('dataset_sintetico_5000_ampliado.csv')
        df = df.drop_duplicates(subset=['texto'])
        
        # Búsqueda completa (Texto + Metadatos)
        df['texto_completo'] = (
            "USUARIO: " + df['usuario'].astype(str) + 
            " | FECHA: " + df['fecha'].astype(str) + 
            " | LIKES: " + df['likes'].astype(str) + 
            " | TEMA: " + df['tema'].astype(str) + 
            " | TEXTO: " + df['texto'].astype(str)
        )
        print(f"   -> {len(df)} textos cargados y procesados.")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit()

def crear_buscador(df):
    print("2. Creando motor de búsqueda matemático (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    matriz_tfidf = vectorizer.fit_transform(df['texto_completo'])
    return vectorizer, matriz_tfidf

def buscar_contexto(query, vectorizer, matriz_tfidf, df, top_k=5):
    query_vec = vectorizer.transform([query])
    similitudes = cosine_similarity(query_vec, matriz_tfidf).flatten()
    indices_top = similitudes.argsort()[-top_k:][::-1]
    resultados = df.iloc[indices_top]['texto_completo'].tolist()
    return resultados

def consultar_llama(pregunta, contexto):
    print("   -> Consultando a Llama 3...")
    
    # --- PROMPT REFORZADO PARA ESPAÑOL ---
    prompt_sistema = """
    Eres un analista de datos y filósofo que habla ÚNICAMENTE ESPAÑOL.
    
    Tus instrucciones son:
    1. Responde SIEMPRE en español.
    2. Usa los datos del CONTEXTO para responder.
    3. Si te preguntan por un dato específico (likes, usuario), búscalo en el contexto y cítalo.
    4. Integra conceptos de Byung-Chul Han, Bauman o Foucault cuando sea pertinente.
    """
    
    prompt_usuario = f"""
    CONTEXTO (DATOS RECUPERADOS):
    {contexto}
    
    PREGUNTA DEL USUARIO:
    {pregunta}
    
    RESPUESTA (EN ESPAÑOL):
    """
    
    completion = client.chat.completions.create(
        model="local-model", 
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.3
    )
    
    return completion.choices[0].message.content

# --- EJECUCIÓN ---
if __name__ == "__main__":
    print("\n--- INICIANDO SISTEMA RAG FINAL ---")
    
    df = cargar_datos()
    vectorizer, matriz = crear_buscador(df)
    
    print("\n>>> SISTEMA LISTO. Escribe 'salir' para cerrar.")
    print(">>> TIP: Si quieres ver los datos crudos, escribe 'ver evidencia' en tu pregunta.\n")
    
    while True:
        pregunta = input("\n>>> Pregunta: ")
        if pregunta.lower() in ['salir', 'exit']:
            break
        
        try:
            # 1. Recuperar
            contexto_lista = buscar_contexto(pregunta, vectorizer, matriz, df)
            contexto_str = "\n---\n".join(contexto_lista)
            
            # --- LÓGICA CONDICIONAL ---
            # Solo muestra el bloque de texto si tú lo pides
            if "ver evidencia" in pregunta.lower() or "debug" in pregunta.lower():
                print("\n" + "-"*40)
                print(f" EVIDENCIA OCULTA ({len(contexto_lista)} fragmentos):")
                print(contexto_str[:600] + "...") 
                print("-" * 40 + "\n")
            # --------------------------
            
            # 2. Generar
            respuesta = consultar_llama(pregunta, contexto_str)
            
            print("\n" + "="*50)
            print(respuesta)
            print("="*50)
            
        except Exception as e:
            print(f"\nERROR: {e}")