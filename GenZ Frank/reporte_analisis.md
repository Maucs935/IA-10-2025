# 📊 ANÁLISIS COMPLETO DEL DATASET

## Información General

**Dataset:** datasetTexto.csv  
**Tamaño:** 198 artículos (se encontraron 2 líneas con errores de formato)  
**Período:** 31 de agosto a 18 de noviembre de 2025 (79 días)  
**Columnas:** ID, Categoria, Titulo, Medio, Fecha, Resumen, Comentario_Reaccion

---

## 1️⃣ DISTRIBUCIÓN POR CATEGORÍA

El dataset está prácticamente equilibrado entre dos temas principales:

- **Generación Z (protestas en México):** 100 artículos (50.5%)
- **Frankenstein (película de Guillermo del Toro):** 98 artículos (49.5%)

**Análisis:** La distribución equitativa sugiere que el dataset fue diseñado intencionalmente para comparar la cobertura mediática de un movimiento social con un evento cultural/cinematográfico.

---

## 2️⃣ ANÁLISIS TEMPORAL

### Distribución por Mes:
- **Agosto 2025:** 2 artículos (ambos de Frankenstein)
- **Octubre 2025:** 12 artículos (todos de Frankenstein)
- **Noviembre 2025:** 184 artículos (84 Frankenstein + 100 Generación Z)

### Hallazgos Temporales:

1. **Pico de actividad en noviembre:** El 92.9% de todos los artículos se concentran en noviembre de 2025, coincidiendo con:
   - Estreno de Frankenstein en Netflix (7 de noviembre)
   - Protestas de la Generación Z (15-18 de noviembre)

2. **Cobertura anticipada:** Los artículos sobre Frankenstein comienzan desde agosto (Festival de Venecia), mostrando una campaña mediática progresiva.

3. **Cobertura reactiva vs. planificada:**
   - **Frankenstein:** Cobertura anticipada y continua (agosto-noviembre)
   - **Generación Z:** Cobertura concentrada y reactiva (solo noviembre)

---

## 3️⃣ MEDIOS DE COMUNICACIÓN

### Estadísticas Generales:
- **Total de medios únicos:** 89 fuentes diferentes
- **Diversidad mediática:** Alta variación entre medios nacionales, internacionales, especializados y generalistas

### Top 10 Medios:
1. **El País** - 12 artículos (principalmente Generación Z)
2. **Milenio** - 6 artículos (mixto)
3. **Reuters** - 4 artículos (Generación Z)
4. **El Financiero** - 4 artículos (Frankenstein)
5. **El Universal** - 4 artículos (Generación Z)
6. **TV Azteca, Euronews, BBC Mundo, Netflix Tudum, Variety** - 2 artículos c/u

### Diferencias por Categoría:

**Generación Z:**
- Dominio de medios mexicanos (El País, Milenio, El Universal)
- Presencia de agencias internacionales (Reuters, Euronews)
- Enfoque en medios de noticias tradicionales

**Frankenstein:**
- Medios especializados en entretenimiento (Variety, Hollywood Reporter)
- Plataformas de streaming (Netflix Tudum)
- Publicaciones culturales (IndieWire, Screen Rant)

---

## 4️⃣ ANÁLISIS DE CONTENIDO

### Longitud de Textos:

**Títulos:**
- Promedio: 48.9 caracteres
- Rango: 26-72 caracteres
- Observación: Títulos concisos y optimizados para compartir en redes sociales

**Resúmenes:**
- Promedio: 55.9 caracteres
- Rango: 38-78 caracteres
- Observación: Muy cortos, tipo "snippet" o extracto breve

**Comentarios/Reacciones:**
- Promedio: 49.7 caracteres
- Observación: Formato similar a tweets o posts de redes sociales

---

## 5️⃣ PALABRAS CLAVE MÁS FRECUENTES

### Top 10 General:
1. **frankenstein** - 30 menciones
2. **toro** (Guillermo del Toro) - 30 menciones
3. **marcha** - 28 menciones
4. **sobre** - 17 menciones
5. **análisis** - 14 menciones
6. **generación** - 12 menciones
7. **protesta** - 12 menciones
8. **gobierno** - 10 menciones
9. **netflix** - 8 menciones
10. **película** - 8 menciones

### Análisis por Categoría:

**Generación Z - Palabras Clave:**
- marcha, protesta, gobierno, jóvenes, generación
- Enfoque: Movimiento social, política, juventud

**Frankenstein - Palabras Clave:**
- frankenstein, toro, película, netflix, director
- Enfoque: Cine, director, plataforma de streaming

---

## 6️⃣ ANÁLISIS DE HASHTAGS

### Estadísticas:
- **Total de hashtags:** 7
- **Hashtags únicos:** 7
- **Promedio por comentario:** 0.04 (muy bajo)

### Hashtags Identificados:

**Generación Z:**
- #MexicoProtesta
- #SomosMas
- #NoSoyUnBot
- #MexicoDespierta
- #ProtestaDigital

**Frankenstein:**
- #GDTFrankenstein
- #CDMX

**Observación:** El uso de hashtags es extremadamente limitado, sugiriendo que los comentarios no son tweets directos o que fueron editados/limpiados.

---

## 7️⃣ HALLAZGOS Y CONCLUSIONES

### 1. Equilibrio Editorial
El dataset muestra un balance perfecto entre cultura pop (cine) y movimiento social (protestas), lo que sugiere un interés en comparar cómo los medios cubren ambos tipos de eventos.

### 2. Temporalidad Concentrada
- **93% de artículos** se publican en noviembre
- Ambos temas alcanzan su pico mediático simultáneamente
- Sugiere que el dataset captura un "momento cultural" específico

### 3. Diversidad de Fuentes
Con 89 medios únicos para 198 artículos, existe alta fragmentación mediática, reflejando el ecosistema actual de medios digitales y especializados.

### 4. Narrativas Distintas

**Frankenstein:**
- Cobertura planificada y promocional
- Enfoque en arte, técnica cinematográfica y crítica
- Tono celebratorio y analítico

**Generación Z:**
- Cobertura reactiva a eventos
- Enfoque en conflicto, demandas y consecuencias
- Tono crítico y político

### 5. Formato de Redes Sociales
La brevedad de títulos, resúmenes y comentarios (todos ~50 caracteres promedio) indica que el dataset está optimizado para consumo en redes sociales, reflejando el ecosistema mediático actual.

### 6. Ausencia de Hashtags
El bajo uso de hashtags (solo 7 en 198 artículos) es anómalo considerando el formato de redes sociales. Posibles explicaciones:
- Los comentarios fueron creados artificialmente o editados
- Son extractos de artículos completos, no posts originales
- Fueron limpiados para análisis

---

## 8️⃣ PREGUNTAS DE INVESTIGACIÓN SUGERIDAS

Este dataset permite explorar:

1. **¿Cómo difiere la cobertura mediática entre eventos culturales y movimientos sociales?**
2. **¿Qué tipo de lenguaje usan los medios para cada categoría?**
3. **¿Existe sesgo político en la cobertura de las protestas?**
4. **¿Cómo se promociona el cine de autor en medios tradicionales vs. especializados?**
5. **¿Qué rol juega la temporalidad en la intensidad de la cobertura?**

---

## 9️⃣ LIMITACIONES DEL DATASET

1. **Pérdida de datos:** 2 filas (1%) no pudieron ser procesadas por errores de formato
2. **Período corto:** Solo 79 días de cobertura
3. **Hashtags limitados:** No refleja el uso real en redes sociales
4. **Sin metadatos adicionales:** Falta información como engagement, alcance, sentimiento

---

## 🎯 RECOMENDACIONES

### Para Análisis Posterior:

1. **Análisis de Sentimiento:** Evaluar el tono positivo/negativo de cada categoría
2. **Modelado de Tópicos:** Usar LDA o NMF para descubrir subtemas ocultos
3. **Análisis de Red:** Mapear relaciones entre medios y temas
4. **Series Temporales:** Predecir picos de cobertura mediática
5. **Clasificación:** Entrenar un modelo para categorizar nuevos artículos automáticamente

---

## 📈 MÉTRICAS FINALES

| Métrica | Valor |
|---------|-------|
| Total de Artículos | 198 |
| Período de Análisis | 79 días |
| Medios Únicos | 89 |
| Categorías | 2 |
| Artículos/Día (promedio) | 2.5 |
| Pico de Publicaciones | 18 nov 2025 |
| Diversidad de Fuentes | Alta (89/198 = 0.45) |

---

**Fecha del Análisis:** Noviembre 2025  
**Analista:** Claude (Anthropic)  
**Herramientas:** Python, Pandas, Matplotlib, Seaborn
