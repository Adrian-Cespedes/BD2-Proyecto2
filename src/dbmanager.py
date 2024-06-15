import pandas as pd

from invertedIndexClass import InvertedIndex
from invertedIndexRAM import InvertedIndexRAM

from mongoClass import Mongo
import time


class DataStoreManager:
    def __init__(self, data_filename, df_headers=[]):
        # ['artist', 'song', 'link', 'text']
        self.data = pd.read_csv(data_filename)
        self.df_headers = df_headers
        self.stores = {
            "inverted_index": InvertedIndex(self.data),
            # "inverted_index_ram": InvertedIndexRAM(self.data),
            "mongo": Mongo(self.data),
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
        if result is None:
            result = pd.DataFrame(columns=self.df_headers)
        else:
            # result = pd.DataFrame(result)
            result = pd.DataFrame(
                [
                    {
                        "artist": self.data.iloc[int(doc_id)]["artist"],
                        "song": self.data.iloc[int(doc_id)]["song"],
                        "lyrics": self.data.iloc[int(doc_id)]["text"],
                        "score": score,
                    }
                    for doc_id, score in result
                ]
            )
            result.columns = self.df_headers
        return result, end_time - start_time
