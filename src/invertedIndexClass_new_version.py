import os
import shutil
import json
import heapq
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
    def __init__(self, file):
        self.path_docs = file
        self.block_size = 100  # Número de documentos por bloque
        self.doc_count = 0 # numero de documentos
        # self.stemmer = SnowballStemmer("spanish")
        self.stemmer = SnowballStemmer("english")
        self.building()
        self.chuks_number = 0

    def build_index(self):
        # SPIMI
        chunk_iter = pd.read_csv(self.path_docs, chunksize=self.block_size)
        # tem dir para almacenar los bloques
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

    def merge_blocks(self, temp_block_dir):
        # Merge blocks into a single index por partes:

        min_heap = []
        json_files = [os.path.join(temp_block_dir, f"block_{i}.json") for i in range(self.chuks_number)]
        file_terms = [self.load_next_term(filename, 1) for filename in json_files]
        file_pointers = [1 for i in range(self.chuks_number)]

        # Initialize heap with the first term from each block
        for i in range(self.chuks_number):
            term = file_terms[i]
            heapq.heappush(min_heap, (list(term.keys())[0], i))

        final_terms = defaultdict(lambda: defaultdict(int))

        index_page = 0
        while min_heap:
            term, i = heapq.heappop(min_heap)

            # extraer los postings de file_terms[i] correctamente
            postings = file_terms[i][term]

            if term in final_terms: 
                final_terms[term].update(postings)
            else:
                final_terms[term] = postings

            # cargar nuevo término del bloque que se uso
            file_pointers[i] += 1
            new_term = self.load_next_term(json_files[i], file_pointers[i])
            if new_term:
                heapq.heappush(min_heap, (list(new_term.keys())[0], i))
                file_terms[i] = new_term

            # Escribir el índice final en disco
            if len(final_terms) >= self.block_size:
                while min_heap:
                    temp_t, temp_i = heapq.heappop(min_heap)
                    if temp_t != term:
                        heapq.heappush(min_heap, (temp_t, temp_i))
                        break

                    # Escribir el índice parcial en disco
                    final_terms[term].update(file_terms[temp_i][temp_t])

                # ordenar los postings por doc_id
                final_terms = {term: dict(sorted(postings.items())) for term, postings in sorted(final_terms.items())}

                # Escribir el índice parcial en disco
                # self.write_json_file(dict(final_terms), os.path.join(temp_index_dir, "index_build.json"))
                index_page += 1
                self.write_file(dict(final_terms), temp_index_dir, index_page)
                final_terms.clear()

        if final_terms:
            final_terms = {term: dict(sorted(postings.items())) for term, postings in sorted(final_terms.items())}
            self.write_json_file(dict(final_terms), os.path.join(temp_index_dir, f"index_{index_page}.json"))

    def load_next_term(self, filename, num_term):
        with open(filename, "r") as file:
            data = json.load(file)

        count = 0
        for key, value in data.items():
            count += 1
            if count == num_term:
                return {key: value}
        return None

    def write_json_file(self, data, filename):
        data_str = json.dumps(data, indent=4)

        # verificar si existe el archivo
        if not os.path.exists(filename): 
            with open(filename, 'w') as file:
                file.write(data_str)
        else:
            with open(filename, 'r+') as file:
                file.seek(0, 2) # mover el puntero al final del archivo

                file.seek(file.tell() - 1)
                file.write(',')
                file.write(data_str[1:-1])  # Eliminar {}
                file.write("}") 

    def write_file(self, data, filename, i):
        with open(
            os.path.join(filename, f"index_{i}.json"), "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def preprocess(self, text):
        tokens = nltk.word_tokenize(text.lower())
        tokens = [w for w in tokens if w not in stoplist and w.isalnum()]
        result = Counter([self.stemmer.stem(w) for w in tokens])
        return result # {word: frequency}

    def building(self):
        self.build_index()
        # self.compute_tf_idf_and_lengths()

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
    dataton = os.path.join(base_path, "spotify_millsongdata_1000.csv")
    index = InvertedIndex(dataton)
    # query1 = "She's just my kind of girl"
    # result = index.retrieve(query1, 5)
    # print(result)
