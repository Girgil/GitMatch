import os
import uvicorn
import sys

from pymongo import MongoClient
from dotenv import load_dotenv
from fastapi import FastAPI

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model.Doc2VecModel import Doc2VecModel
from database.DatabaseManager import DatabaseManager
from ingestion.requests_utils import get_repo, filter_features
from ingestion.preprocessing import preprocess_df

current_directory = os.getcwd()
mlflow_tracking_uri = os.path.join(current_directory, "artifacts", "mlruns")
# Utilisation locale
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri

# cas où on dockerise le serveur mlflow
#os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
production_mlflow_model_readme_uri = 'models:/doc2vec_readme@production'
production_mlflow_model_others_uri = 'models:/doc2vec_others@production'

# Chargement du client mongo de la surcouche pour gérer les interactions avec la base mongo
client = MongoClient('localhost', 27017)
database_manager = DatabaseManager(client['github'])

# Définition des features qu'on considère
features_for_json = [
    'full_name',
    'id',
    'description',
    'language',
    'topics',
    'contents_url',
    'html_url',
    'default_branch',
]

features_for_df = features_for_json + ['owner_id']

# Chargement du modèle
production_model = Doc2VecModel(
    'production',
    production_mlflow_model_readme_uri,
    production_mlflow_model_others_uri,
)

# Chargement du token
env_path = '.env'
load_dotenv(env_path)
token = os.getenv("TOKEN_GITHUB")


# Chargement des représentations vectorielles des répertoires
df_vectors = database_manager.get_df_vectors()

app = FastAPI()

@app.get("/recommend/{owner_name}/{repo_name}")
def recommend(owner_name: str, repo_name: str, k: int):
    name = f'{owner_name}/{repo_name}'
    # Requêter pour récupérer la réponse json
    repo_json = get_repo(token=token, full_name=name)
    
    # Prétraiter pour récupérer un dictionnaire avec readme_preproc et others_preproc
    repo_filtered = filter_features([repo_json], features_for_json)
    df = preprocess_df(
        token=token,
        repos=repo_filtered,
        features=features_for_df,
    )
    
    dict_repo = df.to_dict(orient='records')[0]

    # Récupération des k repos les plus similaires pour chaque représentation
    k_most_similar_with_readme, k_most_similar_with_others = (
        production_model.get_top_k_for_prediction(k, dict_repo, df_vectors)
    )

    # On considère seulement les ids
    ids_with_readme = k_most_similar_with_readme['id'].tolist()
    ids_with_others = k_most_similar_with_others['id'].tolist()

    # Récupération des urls des repos similaires à partir des ids
    readme_urls = database_manager.get_url_list_from_id_list(ids_with_readme)
    others_urls = database_manager.get_url_list_from_id_list(ids_with_others)
    
    return {
        "readme": readme_urls,
        "others": others_urls,
    }

@app.post("/update")
def update_api():
    production_model = Doc2VecModel('production', production_mlflow_model_readme_uri, production_mlflow_model_others_uri)
    df_vectors = database_manager.get_df_vectors()
    return "Mise à jour réussie"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("🚀 FastAPI prêt sur http://localhost:8000")