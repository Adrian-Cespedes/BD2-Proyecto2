import pandas as pd
import joblib
from invertedIndexClass_new_version import InvertedIndex
# from invertedIndexRAM import InvertedIndexRAM
from mongoClass import Mongo
from multimediaIndex import KNN_Secuencial, KNN_RTree, KNN_Faiss
from multimediaFuncs import load_features, reduce_new_input
import librosa
import time


class DataStoreManager:
    # mmfv = MultiMedia Feature Vectors
    def __init__(self, data_filename, df_headers=[], mmfv_filename="", pca_model_filename="", songs_path=""):
        # ['artist', 'song', 'link', 'text']
        self.data = data_filename  # path
        self.mmfv, self.mmfv_song_names = load_features(mmfv_filename)
        # self.mmfv = self.mmfv[:2000, :]
        print("MMFV shape: ", self.mmfv.shape)
        self.df_headers = df_headers
        self.pca = joblib.load(pca_model_filename)
        self.songs_path = songs_path
        self.stores = {
            "inverted_index": InvertedIndex(self.data),
            # "inverted_index_ram": InvertedIndexRAM(self.data),
            "mongo": Mongo(self.data),
            "knn_secuencial": KNN_Secuencial(self.mmfv),
            "knn_rtree": KNN_RTree(self.mmfv),
            "knn_faiss": KNN_Faiss(self.mmfv)
        }
        self.active_store = "inverted_index"

    def set_active_store(self, store_type):
        if store_type in self.stores:
            self.active_store = store_type
            print(f"Active store set to: {store_type}")
        else:
            raise ValueError(
                "Unsupported store type. Choose 'inverted_index' or 'mongo'."
            )

    def retrieve(self, query, k):
        start_time = time.time()
        result = self.stores[self.active_store].retrieve(query, k)
        end_time = time.time()
        # temp
        if result is None or len(result) == 0:
            result = pd.DataFrame(columns=self.df_headers)
        else:
            df = []
            for doc_id, score in result:
                row = pd.read_csv(
                    self.data, skiprows=int(doc_id) + 1, nrows=1, header=None
                ).iloc[0]
                result_dict = {
                    "artist": row.iloc[0],
                    "song": row.iloc[1],
                    "lyrics": row.iloc[3],
                    "score": score,
                }
                df.append(result_dict)
            result = pd.DataFrame(df)

            result.columns = self.df_headers
        return result, end_time - start_time
    
    # def retrieve_media_knn(self, query, k):
    #     start_time = time.time()
    #     query = reduce_new_input(query, self.pca)
    #     result = self.stores[self.active_store].knnSearch(query, k)
    #     # for each tuple (int, int) add the song name
    #     for i in range(len(result)):
    #         result[i] = (self.mmfv_song_names[result[i][0]], result[i][1])
    #     end_time = time.time()
    #     return result, end_time - start_time
    
    # def retrieve_media_knn(self, query, k):
    #     start_time = time.time()
    #     query = extract_segmented_features(query)  # Extraer características segmentadas
    #     if query is not None:
    #         query = reduce_new_segmented_input(query, self.pca)
    #         result = self.stores[self.active_store].knnSearch(query, k)
    #         for i in range(len(result)):
    #             result[i] = (self.mmfv_song_names[result[i][0]], result[i][1])
    #     else:
    #         result = []
    #     end_time = time.time()
    #     return result, end_time - start_time

    def retrieve_media_knn(self, query, k, r, is_knn):
        start_time = time.time()
        query = reduce_new_input(query, self.pca)  # Usar segmentación en la entrada de consulta
        if is_knn:
            result = self.stores[self.active_store].knnSearch(query, k)
        else:
            result = self.stores[self.active_store].rangeSearch(query, r)
        end_time = time.time()
        for i in range(len(result)):
            result[i] = (self.mmfv_song_names[result[i][0]], result[i][1])
        # return every songs_path + song_name + score as tuple in an array
        result = [(self.songs_path + "/" + song_name, song_name, score) for song_name, score in result]
        return result, end_time - start_time

