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
### 2.1 Construcción del índice invertido en memoria secundaria
### 2.2 Ejecución óptima de consultas aplicando Similitud de Coseno
### 2.3 Explicación de la construcción del índice invertido en MongoDB

La creación del indice invertido en MongoDB es relativamente simple, se requiere de una base de datos y una colección. En este caso se utilizo python para toda la configuración y creación:

1. Conexión a la base de datos:
```python
self.client = MongoClient('localhost', 27017)
self.db = self.client[Mongo.db_name]
```
Esta parte se encarga de conectarse a cualquier servidor de MongoDB, para las pruebas de este proyecto se utilizo un servidor local.

2. Creación de la colección:
```python
if Mongo.col_name not in self.db.list_collection_names():
   records = dataton.reset_index().to_dict(orient='records')
   self.db[Mongo.col_name].insert_many(records)
   status = self.db[Mongo.col_name].create_index({[('text', 'text')]})
```

Esta es la parte mas esencial del proyecto, en la cual se crea el indice invertido respecto al atributo `text` de la colección. Para esto se utiliza la función `create_index` de la libreria `pymongo` la cual recibe como parametro el atributo que se desea indexar.(La funcion requiere de minimo 2 atributos para index pero como solo se requiere el atributo `text` es duplicado).

3. Consulta de texto:
```python
def retrieve(self, query, k):
    result = self.db[Mongo.col_name].find({'$text': {'$search': query}}, {'score': {'$meta': 'textScore'}}).sort([('score', {'$meta': 'textScore'})]).limit(k)
    song = [(doc['index'], doc['score']) for doc in result]
    return song
```

Esta funcion permite recibir una query, la cual es una caden de texto sin procesar, y un valor k. El cual busca retornar los k documentos mas relevantes respecto a la query. Para esto la funcion de `find` recibe como parametro la query y el score de relevancia, para luego ordenarlos y limitarlos a k.

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

