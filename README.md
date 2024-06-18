# BD2-Proyecto2
Proyecto 2 para el curso de Base de Datos 2

## Run with conda
```bash
chmod +x ./start.sh
source ./start.sh
```

# Informe del Proyecto

## 1. Introducción
Desarrollar y optimizar algoritmos de búsqueda y recuperación de información basados en contenido, implementando un Índice Invertido para documentos de texto y una estructura multidimensional para imágenes y audio, con el fin de mejorar la eficiencia y precisión en un sistema de recomendación, además de describir el dominio de datos y la relevancia de la indexación en estos contextos.
#### Descripción del Dominio de Datos:
------------------------------------------
#### Importancia de Aplicar Indexación:
   La indexación es crucial para mejorar la eficiencia, precisión y escalabilidad en la búsqueda y recuperación de información en grandes conjuntos de datos textuales y     
  multimedia, optimizando los sistemas de recomendación, además posee las siguientes caracteristicas:

1. **Rapidez en la Recuperación:**
   - **Índice Invertido:** Acceso rápido a documentos con términos específicos.
   - **Estructuras Multidimensionales:** Búsqueda eficiente de imágenes y audios similares.

2. **Reducción de la Complejidad:**
   - **Eficiencia en Consultas:** Consultas en tiempo sublineal, optimizando el uso de memoria y CPU.

3. **Mejora en la Precisión:**
   - **Relevancia de Resultados:** Uso de técnicas de ponderación y ranking como TF-IDF.
   - **Búsqueda Contextual:** Integración de metadatos y características contextuales.

4. **Escalabilidad:**
   - **Manejo de Volumen de Datos:** Sistemas de indexación escalables horizontalmente.
   - **Actualización y Mantenimiento:** Actualizaciones incrementales sin reconstrucción total.

## 2. Backend: Índice Invertido
- Construcción del índice invertido en memoria secundaria
- Ejecución óptima de consultas aplicando Similitud de Coseno
- Explicación de la construcción del índice invertido en PostgreSQL/MongoDB

## 3. Backend: Índice Multidimensional
- Descripción de la técnica de indexación de las librerías utilizadas
- KNN Search y Range Search (si es que lo soporta)
- Análisis de la maldición de la dimensionalidad y estrategias para mitigarlo

## 4. Frontend
- Diseño de la GUI
- Mini-manual de usuario
- Capturas de pantalla de la GUI
- Análisis comparativo visual con otras implementaciones

## 5. Experimentación
- Tablas y gráficos de los resultados experimentales
- Análisis y discusión
- Imágenes o diagramas para una mejor comprensión

