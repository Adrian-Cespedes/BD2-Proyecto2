import os
import json
import nltk
import numpy as np
from collections import defaultdict, Counter
from nltk.stem.snowball import SnowballStemmer
import pandas as pd

current_dir = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(current_dir, os.pardir, "data")
stoplist_path = os.path.join(base_path, "stoplist.txt")
temp_index_dir = os.path.join(current_dir, "temp_indexes")

nltk.download("punkt")
with open(stoplist_path, encoding="utf-8") as file:
    stoplist = [line.rstrip().lower() for line in file]

if not os.path.exists(temp_index_dir):
    os.makedirs(temp_index_dir)


class InvertedIndex:
    def __init__(self, index_file):
        self.index_file = index_file["text"]
        self.block_size = 1000  # Número de documentos por bloque
        self.doc_count = len(self.index_file)
        # self.stemmer = SnowballStemmer("spanish")
        self.stemmer = SnowballStemmer("english")
        self.building()

    def build_index(self):
        for i in range(0, len(self.index_file), self.block_size):
            block = self.index_file[i : i + self.block_size]
            partial_index = defaultdict(lambda: defaultdict(int))
            for doc_id, document in enumerate(block):
                processed = self.preprocess(document)
                for term, tf in processed.items():
                    partial_index[term][i + doc_id] += tf

            # Escribir el índice parcial y las longitudes de documentos en disco
            self.write_partial_index(partial_index, i // self.block_size)

    def write_partial_index(self, partial_index, block_num):
        index_file = os.path.join(temp_index_dir, f"index_{block_num}.json")
        with open(index_file, "w", encoding="utf-8") as file:
            json.dump(partial_index, file, indent=4)

    def preprocess(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [w for w in tokens if w not in stoplist and w.isalnum()]
        result = Counter([self.stemmer.stem(w) for w in tokens])
        return result

    def building(self):
        self.build_index()
        self.compute_tf_idf_and_lengths()

    def compute_tf_idf_and_lengths(self):
        term_doc_count = defaultdict(int)
        tf_idf = defaultdict(lambda: defaultdict(float))
        doc_lengths = defaultdict(float)

        for file in os.listdir(temp_index_dir):
            if file.startswith("index_"):
                with open(
                    os.path.join(temp_index_dir, file), "r", encoding="utf-8"
                ) as f:
                    partial_index = json.load(f)
                    for term, postings in partial_index.items():
                        term_doc_count[term] += len(postings)
                        for doc_id, tf in postings.items():
                            tf_idf[term][doc_id] = (
                                1 + np.log10(tf)
                            ) * self.log_frec_idf(self.doc_count, term_doc_count[term])
                            doc_lengths[doc_id] += tf_idf[term][doc_id] ** 2

        # Convertir las longitudes de los documentos a sus raíces cuadradas
        doc_lengths = {
            doc_id: np.sqrt(length) for doc_id, length in doc_lengths.items()
        }

        # Guardar tf_idf y las longitudes de documentos en disco
        tf_idf_file = os.path.join(temp_index_dir, "tf_idf.json")
        with open(tf_idf_file, "w", encoding="utf-8") as file:
            json.dump(tf_idf, file, indent=4)

        doc_lengths_file = os.path.join(temp_index_dir, "doc_lengths.json")
        with open(doc_lengths_file, "w", encoding="utf-8") as file:
            json.dump(doc_lengths, file, indent=4)

    def log_frec_idf(self, N, df):
        if df > 0:
            return np.log10(N / df)
        return 0

    def retrieve(self, query, k):
        query_vector = self.preprocess(query)
        # query_tf_idf = {term: (1 + np.log10(tf)) for term, tf in query_vector.items()}
        # query_norm = np.sqrt(sum(val**2 for val in query_tf_idf.values()))

        with open(
            os.path.join(temp_index_dir, "tf_idf.json"), "r", encoding="utf-8"
        ) as file:
            tf_idf = json.load(file)

        with open(
            os.path.join(temp_index_dir, "doc_lengths.json"), "r", encoding="utf-8"
        ) as file:
            doc_lengths = json.load(file)

        term_doc_count = {term: len(tf_idf[term]) for term in tf_idf}

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


# Ejemplo de uso
if __name__ == "__main__":
    dataton = pd.read_csv("data/spotify_millsongdata_1000.csv")
    index = InvertedIndex(dataton)
    query1 = "without her"
    result = index.retrieve(query1, 5)
    print(result)
