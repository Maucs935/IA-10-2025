"""
SCRIPT DE ANÁLISIS ADICIONAL - datasetTexto.csv
Este script contiene ejemplos de análisis avanzados que puedes realizar
con el dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

# Cargar el dataset con manejo de errores
df = pd.read_csv('datasetTexto.csv', sep=',', engine='python', on_bad_lines='skip')
df['Fecha'] = pd.to_datetime(df['Fecha'])

print(f"Dataset cargado: {len(df)} artículos")


# ============================================================================
# 2. ANÁLISIS DE SENTIMIENTO (Requiere: pip install textblob)
# ============================================================================

"""
from textblob import TextBlob

def analizar_sentimiento(texto):
    '''Analiza el sentimiento de un texto (-1 negativo, 0 neutral, +1 positivo)'''
    blob = TextBlob(str(texto))
    return blob.sentiment.polarity

# Aplicar análisis de sentimiento
df['sentimiento_titulo'] = df['Titulo'].apply(analizar_sentimiento)
df['sentimiento_resumen'] = df['Resumen'].apply(analizar_sentimiento)
df['sentimiento_comentario'] = df['Comentario_Reaccion'].apply(analizar_sentimiento)

# Comparar sentimiento por categoría
print("\nSentimiento promedio por categoría:")
print(df.groupby('Categoria')[['sentimiento_titulo', 'sentimiento_resumen']].mean())

# Visualizar distribución de sentimiento
plt.figure(figsize=(12, 5))
for i, cat in enumerate(df['Categoria'].unique(), 1):
    plt.subplot(1, 2, i)
    cat_df = df[df['Categoria'] == cat]
    plt.hist(cat_df['sentimiento_titulo'], bins=20, alpha=0.7)
    plt.title(f'Distribución de Sentimiento - {cat}')
    plt.xlabel('Sentimiento')
    plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()
"""


# ============================================================================
# 3. ANÁLISIS DE N-GRAMAS (Palabras y frases más comunes)
# ============================================================================

"""
from sklearn.feature_extraction.text import CountVectorizer

def extraer_ngramas(textos, n=2, top=10):
    '''Extrae los n-gramas más frecuentes'''
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=['de', 'la', 'el', 'en', 'y', 'a'])
    ngrams = vectorizer.fit_transform(textos)
    freq = ngrams.sum(axis=0).A1
    ngram_freq = list(zip(vectorizer.get_feature_names_out(), freq))
    ngram_freq.sort(key=lambda x: x[1], reverse=True)
    return ngram_freq[:top]

# Bigramas más comunes por categoría
for cat in df['Categoria'].unique():
    cat_textos = df[df['Categoria'] == cat]['Titulo'].tolist()
    bigramas = extraer_ngramas(cat_textos, n=2, top=10)
    print(f"\nTop 10 bigramas - {cat}:")
    for bigrama, freq in bigramas:
        print(f"  {bigrama}: {freq}")
"""


# ============================================================================
# 4. CLUSTERING DE TEXTOS (Agrupar artículos similares)
# ============================================================================

"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Crear matriz TF-IDF
vectorizer = TfidfVectorizer(max_features=100, stop_words=['de', 'la', 'el'])
X = vectorizer.fit_transform(df['Titulo'] + ' ' + df['Resumen'])

# Aplicar K-Means
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Ver distribución de clusters
print("\nDistribución de artículos por cluster:")
print(df.groupby(['cluster', 'Categoria']).size().unstack(fill_value=0))

# Ver palabras clave de cada cluster
feature_names = vectorizer.get_feature_names_out()
for i in range(n_clusters):
    cluster_center = kmeans.cluster_centers_[i]
    top_indices = cluster_center.argsort()[-5:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"\nCluster {i}: {', '.join(top_words)}")
"""


# ============================================================================
# 5. MODELADO DE TÓPICOS (LDA)
# ============================================================================

"""
from sklearn.decomposition import LatentDirichletAllocation

# Crear matriz de términos
vectorizer = TfidfVectorizer(max_features=100, stop_words=['de', 'la', 'el'])
X = vectorizer.fit_transform(df['Titulo'] + ' ' + df['Resumen'])

# Aplicar LDA
n_topics = 4
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# Mostrar tópicos
feature_names = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"\nTópico {idx}: {', '.join(top_words)}")
"""


# ============================================================================
# 6. ANÁLISIS DE CO-OCURRENCIA DE PALABRAS
# ============================================================================

"""
from itertools import combinations

def palabras_coocurrentes(textos, top=20):
    '''Encuentra pares de palabras que aparecen juntas frecuentemente'''
    stop_words = {'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'del'}
    
    co_ocurrencias = Counter()
    for texto in textos:
        palabras = [w.lower() for w in re.findall(r'\b\w+\b', texto) 
                   if w.lower() not in stop_words and len(w) > 3]
        co_ocurrencias.update(combinations(sorted(set(palabras)), 2))
    
    return co_ocurrencias.most_common(top)

# Analizar por categoría
for cat in df['Categoria'].unique():
    cat_textos = df[df['Categoria'] == cat]['Titulo'].tolist()
    co_oc = palabras_coocurrentes(cat_textos, top=10)
    print(f"\nPalabras co-ocurrentes - {cat}:")
    for (w1, w2), freq in co_oc:
        print(f"  {w1} + {w2}: {freq}")
"""


# ============================================================================
# 7. ANÁLISIS DE SERIES TEMPORALES
# ============================================================================

def analizar_tendencia_temporal():
    '''Analiza la tendencia temporal de publicaciones'''
    
    # Artículos por día
    diario = df.groupby(['Fecha', 'Categoria']).size().unstack(fill_value=0)
    
    # Calcular media móvil (3 días)
    diario_ma = diario.rolling(window=3).mean()
    
    # Visualizar
    plt.figure(figsize=(14, 6))
    diario_ma.plot(marker='o', linewidth=2)
    plt.title('Tendencia Temporal (Media Móvil 3 días)')
    plt.xlabel('Fecha')
    plt.ylabel('Artículos/día')
    plt.legend(title='Categoría')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Ejecutar
analizar_tendencia_temporal()


# ============================================================================
# 8. ANÁLISIS DE DIVERSIDAD LÉXICA
# ============================================================================

def diversidad_lexica(textos):
    '''Calcula la diversidad léxica (type-token ratio)'''
    todas_palabras = []
    for texto in textos:
        palabras = re.findall(r'\b\w+\b', texto.lower())
        todas_palabras.extend(palabras)
    
    tokens = len(todas_palabras)
    tipos = len(set(todas_palabras))
    
    return tipos / tokens if tokens > 0 else 0

# Comparar diversidad por categoría
for cat in df['Categoria'].unique():
    cat_textos = df[df['Categoria'] == cat]['Titulo'].tolist()
    diversidad = diversidad_lexica(cat_textos)
    print(f"Diversidad léxica {cat}: {diversidad:.4f}")


# ============================================================================
# 9. DETECCIÓN DE TEMAS EMERGENTES
# ============================================================================

def temas_emergentes_por_periodo(df, ventana_dias=7):
    '''Identifica palabras que aumentan su frecuencia en períodos recientes'''
    
    df_sorted = df.sort_values('Fecha')
    fecha_corte = df_sorted['Fecha'].max() - pd.Timedelta(days=ventana_dias)
    
    # Textos recientes vs. anteriores
    recientes = df_sorted[df_sorted['Fecha'] > fecha_corte]['Titulo'].tolist()
    anteriores = df_sorted[df_sorted['Fecha'] <= fecha_corte]['Titulo'].tolist()
    
    # Contar palabras
    stop_words = {'de', 'la', 'el', 'en', 'y', 'a'}
    
    def contar_palabras(textos):
        palabras = []
        for texto in textos:
            palabras.extend([w.lower() for w in re.findall(r'\b\w+\b', texto) 
                           if w.lower() not in stop_words and len(w) > 3])
        return Counter(palabras)
    
    freq_recientes = contar_palabras(recientes)
    freq_anteriores = contar_palabras(anteriores)
    
    # Calcular cambio relativo
    temas = {}
    for palabra in freq_recientes:
        freq_r = freq_recientes[palabra] / len(recientes) if len(recientes) > 0 else 0
        freq_a = freq_anteriores.get(palabra, 0) / len(anteriores) if len(anteriores) > 0 else 0
        if freq_a > 0:
            cambio = (freq_r - freq_a) / freq_a
            if cambio > 0.5:  # Aumento del 50%
                temas[palabra] = cambio
    
    return sorted(temas.items(), key=lambda x: x[1], reverse=True)[:10]

# Ejecutar
print("\nTemas emergentes (últimos 7 días):")
emergentes = temas_emergentes_por_periodo(df, ventana_dias=7)
for palabra, cambio in emergentes:
    print(f"  {palabra}: +{cambio*100:.1f}%")


# ============================================================================
# 10. CLASIFICADOR SIMPLE (Machine Learning)
# ============================================================================

"""
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Preparar datos
vectorizer = TfidfVectorizer(max_features=200)
X = vectorizer.fit_transform(df['Titulo'] + ' ' + df['Resumen'])
y = df['Categoria']

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar clasificador
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluar
y_pred = clf.predict(X_test)
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Predecir categoría de nuevo texto
nuevo_texto = "La película de Del Toro recibe críticas positivas"
nuevo_X = vectorizer.transform([nuevo_texto])
prediccion = clf.predict(nuevo_X)[0]
print(f"\nTexto: '{nuevo_texto}'")
print(f"Categoría predicha: {prediccion}")
"""


# ============================================================================
# NOTAS FINALES
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print("\nPara ejecutar los análisis avanzados, descomenta las secciones relevantes")
print("y asegúrate de instalar las librerías necesarias:")
print("  pip install textblob scikit-learn")
print("\nPara análisis más profundos, considera:")
print("  - spaCy para NLP avanzado")
print("  - transformers para embeddings contextuales")
print("  - networkx para análisis de redes de palabras")
print("="*80)
