import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import json
import pandas as pd

nltk.download("punkt")
with open("stoplist.txt", encoding="utf-8") as file:
    stoplist = [line.rstrip().lower() for line in file]


class InvertIndex:
    def __init__(self, index_file):
        self.index_file = index_file["news"]
        self.index = {}
        self.idf = {}
        self.length = {}
        # build everything
        self.building()

    def build_index(self):
        textos_procesados = []
        for i in range(self.index_file.size):
            textos_procesados.append(self.preproccess(self.index_file[i]))

        for file_index in range(len(textos_procesados)):
            for lex, tf in textos_procesados[file_index].items():
                if lex in self.index:
                    self.index[lex][file_index] = tf
                else:
                    self.index[lex] = {file_index: tf}

    def preproccess(self, texto):
        # 1- tokenizar
        # 2- eliminar signos
        # 3- eliminar stopwords, agregar mas stopwords
        # 4- Aplicar la reduccion de palabras (lexema)
        stemmer = SnowballStemmer("spanish")
        # tokenizar
        tokenized = nltk.word_tokenize(texto.lower())
        # eliminar stopwords
        tokenized = [w for w in tokenized if w not in stoplist and w.isalnum()]
        # lexema
        result = {}
        for w in tokenized:
            w = stemmer.stem(w)
            if w in result:
                result[w] += 1
            else:
                result[w] = 1
        return dict(sorted(result.items()))

    def building(self):
        # Procesar los textos
        # build the inverted index with the collection
        self.build_index()
        temp_tf_idf = {}

        # compute the 1 + log(tf)
        for term, d_tf in self.index.items():
            # term : [(doc,tf)]
            for doc, tf in d_tf.items():
                self.index[term][doc] = self.log_frec_tf(tf)

            # compute idf
            self.idf[term] = self.log_frec_idf(len(self.index_file), len(d_tf))

            # self.length[term] = []
            # for doc, tf in d_tf:
            #    # w_tf * w_idf
            #    self.length[term].append((doc, self.index[term][doc]* self.idf[term]))

            # w_tf * w_idf
            temp_tf_idf[term] = {}
            for doc, tf in d_tf.items():
                temp_tf_idf[term][doc] = self.index[term][doc] * self.idf[term]

                if doc in self.length:
                    self.length[doc] += (self.index[term][doc] * self.idf[term]) ** 2
                else:
                    self.length[doc] = (self.index[term][doc] * self.idf[term]) ** 2

        # compute the length (norm)
        self.length = {doc: tfidf ** (1 / 2) for doc, tfidf in self.length.items()}

        # store in disk
        with open("tf_idf.json", "w") as json_file:
            json.dump(temp_tf_idf, json_file, indent=4)
        with open("doc_norms.json", "w") as json_file:
            json.dump(self.length, json_file, indent=4)

    def retrieval(self, query, k):
        # diccionario para el score
        score = {}
        # preprocesar la query: extraer los terminos unicos
        p_query = self.preproccess(query)

        #########################################
        # calcular el tf-idf del query

        # read tf_idf.json
        with open("tf_idf.json", "r") as json_file:
            tf_idf = json.load(json_file)
        # read doc_norms.json
        with open("doc_norms.json", "r") as json_file:
            doc_norms = json.load(json_file)

        N = self.index_file.size
        norm = 0
        posting_list = {}  # { doc : { term : tf_idf } }

        for term, tf in p_query.items():
            # compute w_tf
            p_query[term] = self.log_frec_tf(tf)

            if term in tf_idf:
                df = len(tf_idf[term])
                p_query[term] *= self.log_frec_idf(N, df)
                norm += p_query[term] ** 2
                for doc, value in tf_idf[term].items():
                    if doc in posting_list:
                        posting_list[doc][term] = value
                    else:
                        posting_list[doc] = {term: value}
            else:
                p_query[term] = 0  # se multiplica tf * idf=0

        norm **= 1 / 2  # sqrt pythonesco

        #########################################
        # aplicar similitud de coseno y guardarlo en el diccionario score
        # score = { doc:score doc1:score ...}
        for doc, term_dict in posting_list.items():
            for term, value in term_dict.items():
                if doc in score:
                    score[doc] += value * p_query[term]
                else:
                    score[doc] = value * p_query[term]

        score = {
            doc: pre_result / (doc_norms[doc] * norm)
            for doc, pre_result in score.items()
        }

        # ordenar el score de forma descendente
        result = sorted(score.items(), key=lambda tup: tup[1], reverse=True)
        # retornamos los k documentos mas relevantes (de mayor similitud al query)
        return result[:k]

    # HELPERS
    def log_frec_tf(self, tf):
        if tf > 0:
            return 1 + np.log10(tf)
        return 0

    def log_frec_idf(self, N, df):
        if df > 0:
            return np.log10(N / df)
        return 0


# dataton = pd.read_csv("df_total.csv")
# dataton.head()
# index = InvertIndex(dataton)
#
# query1 = "Marcelo se quedó jato"
# result = index.retrieval(query1, 10)
# print(result)


def test_func(query, k):
    dataton = pd.read_csv("df_total.csv")
    index = InvertIndex(dataton)
    result = pd.DataFrame(index.retrieval(query, k))
    result.columns = ["ID", "Score"]
    return result
