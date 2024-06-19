from pymongo import MongoClient
import pandas as pd
from pprint import pprint

class Mongo:
    db_name = 'DB_A'
    col_name = 'data_A'
    def __init__(self, data):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client[Mongo.db_name]

        #Crear indice y rellenarlo si no existe
        if Mongo.col_name in self.db.list_collection_names():
            #delete collection
            self.db[Mongo.col_name].drop()
            print("MongoDB collection deleted")
        if Mongo.col_name not in self.db.list_collection_names():
            records = data.reset_index().to_dict(orient='records')
            status = self.db[Mongo.col_name].create_index([('text', 'text')])
            self.db[Mongo.col_name].insert_many(records)
            print("MongoDB collection created and filled")
            print("Index status: ", status)

    def retrieve(self, query, k):
        result = self.db[Mongo.col_name].find({'$text': {'$search': query}}, {'score': {'$meta': 'textScore'}}).sort([('score', {'$meta': 'textScore'})]).limit(k)
        #result = self.db[Mongo.col_name].find({'$text': {'$search': query}}).aggregate({'$addFields': {'score': {'$meta': 'textScore'}}})
        #result = list(result)
        #result.sort(key=lambda x: x['score'], reverse=True)
        song = [(doc['index'], doc['score']) for doc in result]
        return song
    
    def explain_retrieve_time_millis(self, query, k):
        result = self.db[Mongo.col_name].find({'$text': {'$search': query}}, {'score': {'$meta': 'textScore'}}).sort([('score', {'$meta': 'textScore'})]).limit(k)
        explain_result = result.explain()
        print(explain_result.get('executionStats').get('executionStages').get('executionTimeMillisEstimate'))
        return explain_result.get('executionStats').get('executionTimeMillis')



if __name__ == "__main__":
    #Test de tiempos
    query = "She is mhy type of girl"
    
    results = {}
    k = 5
    N = [1000,2000,4000,8000,16000,32000,64000,128000,230600]
    for i in N:
        print2 = f"Query: {query} - N: {i}"
        print(print2)
        data = pd.read_csv(f"data/spotify_millsongdata_{i}.csv")
        mongo = Mongo(data)
        results[i] = mongo.explain_retrieve_time_millis(query, k) / 1000 #comvertir a segundos
    print(results)
        

