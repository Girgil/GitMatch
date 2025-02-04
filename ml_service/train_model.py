from pymongo import MongoClient
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import pickle

mongo_client = MongoClient("mongodb://mongo:27017/")
db = mongo_client["github"]
collection = db["repos"]

repos = list(collection.find({}, {"name": 1, "description": 1}))
documents = [TaggedDocument(words=(repo["description"] or "").split(), tags=[repo["name"]]) for repo in repos]

model = Doc2Vec(documents, vector_size=50, window=2, min_count=1, workers=4)

import pickle

# save
with open('/data/doc2vec.pk','wb') as f:
    pickle.dump(model,f)

# model.save("/data/doc2vec.model")

print("🎯 Modèle entraîné et sauvegardé")
