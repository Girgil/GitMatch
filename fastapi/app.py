from fastapi import FastAPI
from gensim.models.doc2vec import Doc2Vec
import pickle

app = FastAPI()
# model = Doc2Vec.load("/data/doc2vec.model")  # Chemin vers le modèle sauvegardé


# load
with open('/data/doc2vec.pk', 'rb') as f:
    model: Doc2Vec = pickle.load(f)

@app.get("/recommend/{repo_name}")
def recommend(repo_name: str):
    similar_repos = model.dv.most_similar(repo_name, topn=5)
    return {"recommendations": similar_repos}

@app.get("/hello")
def hello():
    return {"hello": "world"}

print("🚀 FastAPI prêt sur http://localhost:8000")