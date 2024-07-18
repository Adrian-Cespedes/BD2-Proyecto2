# BD2-Proyecto2
Proyecto 2 para el curso de Base de Datos 2

**Integrantes**
- Chincha León, Marcelo Andres
- David Manfredo Herencia Galván
- Adrian Joshep Céspedes Zevallos
- Gabriel Eduardo Blanco Gutierrez

## Run with conda
```bash
chmod +x ./start.sh
source ./start.sh
```

# Informe del Proyecto

## 1. Introducción
Desarrollar y optimizar algoritmos de búsqueda y recuperación de información basados en contenido, implementando un Índice Invertido para documentos de texto y una estructura multidimensional para imágenes y audio, con el fin de mejorar la eficiencia y precisión en un sistema de recomendación, además de describir el dominio de datos y la relevancia de la indexación en estos contextos.

Aparte se añadio metodos y tecnicas de busqueda para busqueda de archivos multimedia, especificamente audio. Utilizando un data set de 2000 canciones se implemento un sistema de busqueda por similitud de audio, utilizando la libreria `librosa` para extraer las caracteristicas de las canciones, `rtree`  y  `FAISS` para la busqueda de los vecinos mas cercanos.

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
   - **Manejo de Volumen de Datos:** Sistema de indexación escalables en memoria secundaria.

## 2. Backend

### 2.1 Índice Invertido
#### 2.1.1 Construcción del índice invertido en memoria secundaria
En el indice invertido se opto por usar SPIMI, con la optimizaciones de :
-  **Buffer de bloques** : Se utilizo un buffer de bloques para la escritura de los bloques en memoria secundaria, esto con el fin de reducir la cantidad de escrituras y accesos en disco.
-  **Terminos directos** : Se evita utilizar un diccionario para almacenar los terminos, en su lugar se utilizan los docID directamente en el bloque. Esto con el fin de reducir el uso de memoria y la cantidad de accesos a disco.

La construcción del indice invertido con estas optimizaciones se realiza en 3 pasos:
1. **Preprocesamiento:** Se tokeniza y normaliza el texto de los documentos, eliminando signos de puntuación, caracteres especiales y stopwords.

```python
def preprocess(self, text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in stoplist and w.isalnum()]
    result = Counter([self.stemmer.stem(w) for w in tokens])
    return result # {word: frequency}
```

2. **Se construye el indice invertido:** Despues de la preprocesacion se construye el indice, el cual siguiendo la técnica de SPIMI se divide en bloques y se escriben en memoria secundaria. En este caso, se crea un diccionario 
```python
def build_index(self):
       # SPIMI
       chunk_iter = pd.read_csv(self.path_docs, chunksize=self.block_size)
       temp_block_dir = os.path.join(temp_index_dir, "blocks")
       if not os.path.exists(temp_block_dir):
           os.makedirs(temp_block_dir)
       for i, chunk in enumerate(chunk_iter):
           partial_index = defaultdict(lambda: defaultdict(int))
           len_chunk = len(chunk["text"])
           for doc_id, document in enumerate(chunk["text"]):
               processed = self.preprocess(document)
               for term, tf in processed.items():
                   # almacenar {term: {doc_id: tf}}
                   if term not in partial_index:
                       partial_index[term] = {(i * len_chunk) + doc_id: tf}
                   else:
                       partial_index[term][doc_id] = tf
           # ordenar el índice parcial por término  y cada lista de postings por doc_id
           partial_index = {term: dict(sorted(postings.items())) for term, postings in sorted(partial_index.items())}
           # Escribir el índice parcial en memoria secundaria as a JSON file
           with open( os.path.join(temp_block_dir, f"block_{i}.json"), "w", encoding="utf-8") as file:
               json.dump(partial_index, file, indent=4)
           self.doc_count += len_chunk
           self.chuks_number = i + 1
       self.merge_blocks(temp_block_dir)
       # delete blocks carpeta entera
       if os.path.exists(temp_block_dir):
           shutil.rmtree(temp_block_dir)
```

3. **Merge de Bloques:** Se realiza el merge pero como los indices ya se guardan ordenados por terminos, la construccion de manera ordenada por terminos y docID se realiza de manera eficiente.


#### 2.2.2 Ejecución óptima de consultas aplicando Similitud de Coseno
Para la ejecución optima de consultas utilizando similitud de coseno se utilizo la siguiente logica:

1. Obtener el vector de consulta y normalizarlo.
2. Buscar los documentos que contienen los terminos de la consulta. (Cargar los bloques que contienen los terminos de la consulta)
3. Calcular el score de similitud de coseno entre la consulta y los documentos.

La funcion retrieve contiene la siguiente parte de codigo:
```python
...
query_tf_idf = {
     term: (1 + np.log10(tf))
     * self.log_frec_idf(self.doc_count, term_doc_count.get(term, 0))
     for term, tf in query_vector.items()
}
query_norm = np.sqrt(sum(val**2 for val in query_tf_idf.values()))
scores = defaultdict(float)
for term in query_tf_idf:
    if term in tf_idf:
        for doc_id, tf_idf_val in tf_idf[term].items():
            scores[doc_id] += query_tf_idf[term] * tf_idf_val
for doc_id in scores:
    if doc_id in doc_lengths:
        scores[doc_id] /= query_norm * doc_lengths[doc_id]
sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
return sorted_scores[:k] if sorted_scores else None
...
```
Esto permite leer los  bloques que contienen


#### 2.2.3 Explicación de la construcción del índice invertido en MongoDB

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

### 2.2 Busqueda Multimedia

Para la busqueda de archivos multimedia se utilizo la estructura de datos RTREE, la cual posee una implementacion en python llamada `rtree`. Esta estructura de datos permite realizar busquedas de rango y knn de manera eficiente. Aunque tambien se comparo con otra liberia llamada `FAISS` desarollada por facebook, esta libreria permite utilizar varios indices multidiemnsionales, pero en este caso se utilizo solo el indice de `IndexFlatL2`.

#### 2.2.1 KNN Search

La función de busqueda por KNN consiste en obtener los vecinos mas cercanos a un punto de inicio, en este caso seria la canción a consultar. El algoritmo posee la siguiente logica:

```python
funcion KNN_Search(indice_datos, query, k):

    c_dist = []
    for p in indice_datos:
        distance = euclidean_distance(point, query_point)
        Añadir (point, distance) a c_dist U
    
    Ordenar c_dist en base a la distancia (de menor a mayor)
    
    n_vecinos = []
    for i in range(k):
        Añadir c_dist[i][0] a n_vecinos
    
    return n_vecinos
```

Este algoritmo puede ser modificado para trabajar con diversos indices, como lo serian los indices de `FAISS` o `RTREE`, aunque tambien se implementó una versión que no usa indices (Sequential Search), con el fin de comparar los tiempos de ejecución.

Aqui se muestra un ejemplo de KNN Search en ejecución:
![image](https://upload.wikimedia.org/wikipedia/commons/d/db/Nearest-neighbor_chain_algorithm_animated.gif)


#### 2.2.2 Range Search 

La función de busqueda por rango consiste en obtener todos los puntos que se encuentran dentro de un rango especificado.  El algoritmo posee la siguiente logica:

```python
funcion Range_Search(indice_datos, query, r):

    n_vecinos = []
    for p in indice_datos:
        distance = euclidean_distance(point, query_point)
        if distance <= r:
            Añadir point a n_vecinos
    
    return n_vecinos
```

Asi como en el caso de KNN Search, este algoritmo puede ser modificado para trabajar con diversos indices,(FAISS, RTREE) o sin indices (Sequential Search).

Aqui se muestra un ejemplo de Range Search en ejecución:
![image](https://upload.wikimedia.org/wikipedia/commons/9/9b/Range_search_animation.gif)

## 3. Frontend
![image](https://github.com/Adrian-Cespedes/BD2-Proyecto2/assets/130480550/bd6f6a63-1698-4bbb-ad50-cdc8142acfa8)

Para la implementación de la interfaz de usuario, se utilizó Gradio, una herramienta para crear interfaces web interactivas de manera rápida y sencilla. A continuación, se muestran las características principales de la interfaz desarrollada para el proyecto:

1. Interfaz Interactiva para Búsqueda de Consultas:

* Ingreso de Consultas: La interfaz permite al usuario ingresar la consulta que desea buscar. 
* Selección de Técnica de Recuperación: Es posible elegir entre dos técnicas de recuperación de información:
MongoDB: Utilizando la indexación de texto proporcionada por MongoDB.
Implementación Propia: Utilizando el índice invertido desarrollado en este proyecto.
* Selección del Número de Documentos (k): Permite seleccionar de manera sencilla el número de documentos que se desea recuperar, es decir, el valor de k para obtener "el top k".
  
2. Resultados de la Consulta:

* Visualización de Resultados: Una vez procesada la consulta, la interfaz muestra los resultados recuperados, proporcionando una lista de los documentos más relevantes.
* Tiempo de Procesamiento: En la parte inferior de la interfaz, se muestra el tiempo que tomó realizar la consulta, ofreciendo una referencia sobre la eficiencia del sistema.

## 4. Experimentación
- Tablas y gráficos de los resultados experimentales

|       Palabras     |      Mongo(sec)   | Inverted Index (sec) |
|--------------------|-------------------|----------------------|
| 1000               |        0.0056     |     0.0427           |
| 2000               |        0.0058     |     0.0787           |
| 4000               |        0.0106     |     0.0915           |
| 8000               |        0.0256     |     0.2028           |
| 16000              |        0.0332     |     0.4128           |


![image](https://github.com/Adrian-Cespedes/BD2-Proyecto2/assets/130480550/201c504b-2e86-4a6d-a477-54a3e9703c9f)

La gráfica compara los tiempos de búsqueda entre dos métodos diferentes: MongoDB y el Índice Invertido, en función del número de documentos. Aquí se presenta un análisis conciso de los resultados:

Desempeño de MongoDB (línea azul):

Consistencia: Los tiempos de búsqueda se mantienen bajos y estables a medida que aumenta el número de documentos.
Eficiencia: MongoDB maneja eficientemente búsquedas en conjuntos de datos grandes, con tiempos de respuesta mínimos incluso con 16000 documentos.
Desempeño del Índice Invertido (línea naranja):

Incremento Progresivo: Los tiempos de búsqueda aumentan considerablemente a medida que crece el número de documentos.
Escalabilidad Limitada: El incremento lineal en el tiempo de búsqueda sugiere que el Índice Invertido es menos eficiente para conjuntos de datos grandes en comparación con MongoDB.

Comparación General:

Superioridad de MongoDB: En todos los puntos de datos, MongoDB muestra tiempos de búsqueda significativamente más rápidos.
Necesidad de Optimización: El Índice Invertido podría requerir optimizaciones adicionales para mejorar su desempeño en grandes volúmenes de datos.

Conclusión sobre los resultados obtenidos
MongoDB demuestra ser más eficiente y escalable para búsquedas en grandes conjuntos de documentos en comparación con el Índice Invertido, que muestra un aumento considerable en el tiempo de búsqueda a medida que crece el número de documentos.



