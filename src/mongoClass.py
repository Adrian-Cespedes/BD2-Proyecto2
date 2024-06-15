from pymongo import MongoClient
import pandas as pd
class Mongo:
    db_name = 'DB_A'
    col_name = 'data_A'
    def __init__(self, data):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client[Mongo.db_name]

        #Crear indice y rellenarlo si no existe
        if Mongo.col_name not in self.db.list_collection_names():
            records = dataton.reset_index().to_dict(orient='records')
            self.db[Mongo.col_name].insert_many(records)
            status = self.db[Mongo.col_name].create_index({[('text', 'text')]})
            print("MongoDB collection created and filled")
            print("Index status: ", status)

    def retrieve(self, query, k):
        result = self.db[Mongo.col_name].find({'$text': {'$search': query}}, {'score': {'$meta': 'textScore'}}).sort([('score', {'$meta': 'textScore'})]).limit(k)
        #result = self.db[Mongo.col_name].find({'$text': {'$search': query}}).aggregate({'$addFields': {'score': {'$meta': 'textScore'}}})
        #result = list(result)
        #result.sort(key=lambda x: x['score'], reverse=True)
        song = [(doc['index'], doc['score']) for doc in result]
        return song
                                      
if __name__ == "__main__":
    dataton = pd.read_csv("data/spotify_millsongdata_1000.csv")
    mongo = Mongo(dataton)
    query1 = "She's just my kind of girl"
    result = mongo.retrieve(query1, 5)
    print(result)

