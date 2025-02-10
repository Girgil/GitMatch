from pymongo import MongoClient
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import pickle

mongo_client = MongoClient("mongodb://mongo:27017/")
db = mongo_client["repertoires_db"]
rep_features = db["repertoires_features"]
rep_vecs = db["repertoires_vecs"]

repos_features = list(rep_features.find({}, {"id": 1, "readme_preproc": 1, "others_preproc": 1, "url": 1}))
documents = [TaggedDocument(words=(repos_features["readme_preproc"] or "").split(), tags=[repo["readme_preproc"]]) for repo in repos_features]

model = Doc2Vec(documents, vector_size=50, window=2, min_count=1, workers=4)

# save
with open('/data/doc2vec.pk','wb') as f:
    pickle.dump(model,f)

# model.save("/data/doc2vec.model")

print("🎯 Modèle entraîné et sauvegardé")
