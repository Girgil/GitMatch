import mlflow
import os
import pandas as pd
import numpy as np
import sys

from src.model.Doc2VecModel import Doc2VecModel
from src.model.Doc2VecTrainer import Doc2VecTrainer
from src.model.Evaluation import Evaluation
from mlflow.exceptions import MlflowException
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from src.database.DatabaseManager import DatabaseManager
from pymongo import MongoClient
from src.model.MLFlowManager import MLFlowManager

def check_if_model_exists(model_uri):
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return True
    except MlflowException as e:
        return False

# Vérifier si un modèle est en staging/production
# --> apprendre si ce n'est pas le cas un modèle pour chacun

def load_model(
    staging,
    is_model_registered,
    database_manager,
    mlflow_model_readme_uri,
    mlflow_model_others_uri
):
    model = None
    
    train_df, test_df = database_manager.get_train_test_split_features()

    if is_model_registered:
        print("on charge le modèle")
        model = Doc2VecModel(
            staging=staging,
            mlflow_readme_model_uri=mlflow_model_readme_uri,
            mlflow_others_model_uri=mlflow_model_others_uri,
        )
    else:
        # Sinon on l'apprend
        print("on doit apprendre le modèle")
        doc2vec_trainer = Doc2VecTrainer(staging=staging)
        train_vectors, test_vectors = doc2vec_trainer.train(train_df.iloc[:2], test_df.iloc[:2])
        database_manager.insert_repos_vectors_for_one_stage(train_vectors, staging)
        database_manager.insert_repos_vectors_for_one_stage(test_vectors, staging)
        model = Doc2VecModel(
            staging=staging,
            mlflow_readme_model_uri=mlflow_model_readme_uri,
            mlflow_others_model_uri=mlflow_model_others_uri,
        )
        
    return model

def ensure_every_repo_is_vectorised(
    database_manager,
    staging_model,
    production_model
):
    # Inférence des nouvelles données
    # Récupération des repos qui n'ont pas encore toutes leurs représentations vectorielles
    df_repos_features_to_predict = database_manager.get_repos_without_all_vectors()

    # Si des répertoires ont besoin d'être inféré on infère
    if len(df_repos_features_to_predict) > 0:
        oriented_df_repos_features_to_predict = df_repos_features_to_predict.to_dict(orient='records')

        stg_vectors = []
        prd_vectors = []
        
        for features in oriented_df_features_repos:
            stg_vectors.append(staging_model.predict(features))
            prd_vectors.append(production_model.predict(features))

        vectors = []
        
        for ii, _ in enumerate(stg_vectors):
            vectors_dict = {
                'id': stg_vectors[ii]['id'],
                'staging_readme_vector': stg_vectors[ii]['staging_readme_vector'],
                'staging_others_vector': stg_vectors[ii]['staging_others_vector'],
                'production_readme_vector': prd_vectors[ii]['production_readme_vector'],
                'production_others_vector': prd_vectors[ii]['production_others_vector'],
            }
        
            vectors.append(vectors_dict)

        cols = [
            'id',
            'staging_readme_vector',
            'staging_others_vector',
            'production_readme_vector',
            'production_others_vector',
        ]
        
        df_vectors_to_insert = pd.DataFrame(
            vectors,
            columns=cols
        )

        database_manager.insert_repos_vector_for_both_stages(df_vectors_to_insert)

def compare_models(
    staging_model,
    production_model,
    database_manager,
):
    # Evaluation des deux modèles
    evaluation = Evaluation(staging_model, production_model)
    
    df_vectors = database_manager.get_df_vectors()
    staging_vectors, production_vectors = database_manager.get_df_split_staging_production(df_vectors)

    users_random_chosen_vectors, users_remaining_test_vectors = database_manager.get_users_test_vectors_splitted()

    stg_scores, prd_scores = evaluation.evaluate(
        staging_vectors=staging_vectors,
        production_vectors=production_vectors,
        test_chosen_vectors=users_random_chosen_vectors,
        test_remaining_vectors=users_remaining_test_vectors,
        k=10
    )

    comparison_score = stg_scores > prd_scores
    nb_better_stg_scores = np.count_nonzero(comparison_score)

    return nb_better_stg_scores

def take_action_according_to_comparison_score(
    nb_better_stg_scores,
    database_manager,
):
    mlflow_manager = MLFlowManager()
    
    if nb_better_stg_scores > 0:
        # Archive modèle en production
        print('Archivage production')
        mlflow_manager.archive_model_from_production("doc2vec_readme")
        mlflow_manager.archive_model_from_production("doc2vec_others")

        print('Promotion staging')
        # Promut le modèle en staging en production
        mlflow_manager.promote_to_production("doc2vec_readme")
        mlflow_manager.promote_to_production("doc2vec_others")

        print('Passage des vecteurs de staging en production')
        # Transfère les vecteurs de staging en production
        database_manager.upgrade_staging_vectors()
        
        # Requêter pour que l'api mette à jour son modèle et ses vecteurs
        
    else:
        print('Archivage staging')
        # Archive modèle en staging
        mlflow_manager.archive_model_from_staging("doc2vec_readme")
        mlflow_manager.archive_model_from_staging("doc2vec_others")

    print('Apprentissage staging')
    # Apprend un nouveau modèle de staging
    doc2vec_trainer = Doc2VecTrainer(staging='staging')
    train_df, test_df = database_manager.get_train_test_split_features()
    train_vectors, test_vectors = (
        doc2vec_trainer.train(train_df.iloc[:3], test_df.iloc[:3])
    )

    print('Enregistrement des nouveaux vecteurs de staging')
    # Enregistre les nouveaux vecteurs en staging
    '''
    database_manager.insert_repos_vectors_for_one_stage(
        train_vectors,
        'staging'
    )
    database_manager.insert_repos_vectors_for_one_stage(
        test_vectors, 
        'staging'
    )
    '''

def sequential_jobs(
    database_manager,
    staging_mlflow_model_readme_uri,
    staging_mlflow_model_others_uri,
    production_mlflow_model_readme_uri,
    production_mlflow_model_others_uri,
):
    print('Chargement des modèles')
    is_staging_model_registered = check_if_model_exists(staging_mlflow_model_readme_uri)
    is_production_model_registered = check_if_model_exists(production_mlflow_model_readme_uri)

    staging_model = load_model(
        staging='staging',
        is_model_registered=is_staging_model_registered,
        database_manager=database_manager,
        mlflow_model_readme_uri=staging_mlflow_model_readme_uri,
        mlflow_model_others_uri=staging_mlflow_model_others_uri,
    )
    
    production_model = load_model(
        staging='production',
        is_model_registered=is_production_model_registered,
        database_manager=database_manager,
        mlflow_model_readme_uri=production_mlflow_model_readme_uri,
        mlflow_model_others_uri=production_mlflow_model_others_uri,
    )
    print('Inférence des nouveaux répertoires')
    # Inférence des nouveaux répertoires
    ensure_every_repo_is_vectorised(database_manager, staging_model, production_model)

    print('Évaluation des deux modèles')
    # Comparaison des scores
    nb_better_stg_scores = compare_models(staging_model, production_model, database_manager)
    print(f'Le modèle de staging a {nb_better_stg_scores} meilleurs scores')

    print('Prise de décision')
    # Prise de décision
    take_action_according_to_comparison_score(nb_better_stg_scores, database_manager)

    print('Fin du job')
        

def main():
    #os.environ["MLFLOW_TRACKING_URI"] = "/artifacts/mlruns"

    # Obtenir le chemin absolu du répertoire courant
    current_directory = os.getcwd()
    
    # Construire le chemin pour MLFLOW_TRACKING_URI
    mlflow_tracking_uri = os.path.join(current_directory, "artifacts", "mlruns")
    
    # Définir la variable d'environnement MLFLOW_TRACKING_URI
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    
    client = MongoClient('localhost', 27017)
    database_manager = DatabaseManager(client['github'])

    # Créer un planificateur
    scheduler = BlockingScheduler()
    
    staging_mlflow_model_readme_uri = 'models:/doc2vec_readme@staging'
    staging_mlflow_model_others_uri = 'models:/doc2vec_others@staging'
    
    production_mlflow_model_readme_uri = 'models:/doc2vec_readme@production'
    production_mlflow_model_others_uri = 'models:/doc2vec_others@production'
    '''
    sequential_jobs(
        database_manager=database_manager,        
        staging_mlflow_model_readme_uri=staging_mlflow_model_readme_uri,
        staging_mlflow_model_others_uri=staging_mlflow_model_others_uri,
        production_mlflow_model_readme_uri=production_mlflow_model_readme_uri,
        production_mlflow_model_others_uri=production_mlflow_model_others_uri,
    )
    '''
    # Planifier les fonctions pour s'exécuter tous les jours à 9h
    scheduler.add_job(
        sequential_jobs,
        CronTrigger(hour=15, minute=19, second=10),
        args=[
            database_manager,
            staging_mlflow_model_readme_uri,
            staging_mlflow_model_others_uri,
            production_mlflow_model_readme_uri,
            production_mlflow_model_others_uri,
        ]
    )
    print("Planificateur démarré...")

    # Démarrer le planificateur
    scheduler.start()

if __name__ == "__main__":
    main()
