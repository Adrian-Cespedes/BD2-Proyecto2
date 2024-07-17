import heapq
from rtree import index
import faiss    
import numpy as np

class KNN_Secuencial:
    def __init__(self, collection):
        self.collection = collection

    def knnSearch(self, query, k):
        np_query = np.array(query)
        max_heap = []

        # for idx, row in self.collection.iterrows():
        #     dist = np.linalg.norm(np.array(row) - np_query)
        #     if len(max_heap) < k:
        #         heapq.heappush(max_heap, (-dist, idx))
        #     else:
        #         if -dist > max_heap[0][0]:
        #             heapq.heapreplace(max_heap, (-dist, idx))

        for i in range(self.collection.shape[0]):
            dist = np.linalg.norm(np.array(self.collection[i]) - np_query)
            if len(max_heap) < k:
                heapq.heappush(max_heap, (-dist, i))
            else:
                if -dist > max_heap[0][0]:
                    heapq.heapreplace(max_heap, (-dist, i))
        
        top_k = [(idx, -dist) for dist, idx in sorted(max_heap, key=lambda x: -x[0])]
        return top_k

    def rangeSearch(self, query, r):
        np_query = np.array(query)
        results = []

        for idx, row in self.collection.iterrows():
            dist = np.linalg.norm(np.array(row) - np_query)
            if dist <= r:
                results.append((idx, dist))
        
        return results
    
class KNN_RTree:
    def __init__(self, collection):
        self.collection = collection
        p = index.Property()
        p.dimension = collection.shape[1]
        self.idx = index.Index(properties=p)
        # for idx, row in collection.iterrows():
        #     point = tuple(row)
        #     # Crear una bounding box para puntos 3D
        #     bounding_box = point + point  # (minx, miny, minz, maxx, maxy, maxz)
        #     self.idx.insert(idx, bounding_box)

        for i in range(self.collection.shape[0]):
            point = tuple(self.collection[i])
            bounding_box = point + point  # (minx, miny, minz, maxx, maxy, maxz)
            self.idx.insert(i, bounding_box)
        
    def knnSearch(self, query, k):
        query_point = tuple(query)
        bounding_box = query_point + query_point
        nearest = list(self.idx.nearest(bounding_box, k))
        print("k = ", 1)
        print("nearest", nearest)
        results = [(i, np.linalg.norm(np.array(self.collection.iloc[i]) - np.array(query))) for i in nearest]
        return sorted(results, key=lambda x: x[1])
    
    def rangeSearch(self, query, r):
        query_point = tuple(query)
        bounding_box = query_point + query_point
        results = []
        for idx in self.idx.intersection(bounding_box):
            dist = np.linalg.norm(np.array(self.collection.iloc[idx]) - np.array(query))
            if dist <= r:
                results.append((idx, dist))
        return results
    
class KNN_Faiss:
    def __init__(self, collection):
        self.collection = collection
        self.d = collection.shape[1]  # Dimensionalidad de los datos
        self.index = faiss.IndexFlatL2(self.d)  # Índice L2 para distancias euclidianas
        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)  # Mover el índice a la GPU
        self.index.add(collection.astype(np.float32))
        
    def knnSearch(self, query, k):
        query = np.array(query).astype(np.float32).reshape(1, -1)  # Asegurar que la consulta esté en el formato correcto
        distances, indices = self.index.search(query, k)
        results = [(indices[0][i], distances[0][i]) for i in range(k)]
        return results
    
    def rangeSearch(self, query, r):
        query = np.array(query).astype(np.float32).reshape(1, -1)  # Asegurar que la consulta esté en el formato correcto
        distances, indices = self.index.search(query, self.collection.shape[0])
        results = [(indices[0][i], distances[0][i]) for i in range(self.collection.shape[0]) if distances[0][i] <= r**2]
        return results

# Ejemplo de uso
# print("KNN-Secuencial")

# data = {
#     'feature1': [1, 2, 3],
#     'feature2': [4, 5, 6],
#     'feature3': [7, 8, 9]
# }
# collection = pd.DataFrame(data)

# query = [2, 5, 8]

# knn = KNN_Secuencial(collection)
# k = 2
# r = 999999

# knn_result = knn.knnSearch(query, k)
# range_result = knn.rangeSearch(query, r)

# print("KNN Result:", knn_result)
# print("Range Search Result:", range_result)

# ##################################

# print("\nKNN-RTree")

# knn_rtree = KNN_RTree(collection)

# knn_result = knn_rtree.knnSearch(query, k)
# range_result = knn_rtree.rangeSearch(query, r)

# print("KNN Result:", knn_result)
# print("Range Search Result:", range_result)

# ##################################

# print("\nKNN-Faiss")

# knn_faiss = KNN_Faiss(collection)

# knn_result = knn_faiss.knnSearch(query, k)
# range_result = knn_faiss.rangeSearch(query, r)

# print("KNN Result:", knn_result)
# print("Range Search Result:", range_result)