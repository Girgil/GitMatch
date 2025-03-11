import mlflow
import pickle
import pandas as pd
import numpy as np
import os
import sys

from mlflow.tracking import MlflowClient
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model.MLFlowManager import MLFlowManager
from model.Doc2VecWraper import Doc2VecWraper

class Doc2VecTrainer():

    def __init__(self, staging):
        """
        staging: staging ou production
        """

        self._staging = staging

        self._doc2vec_readme = None
        self._doc2vec_others = None

    def _df_to_documents_list(self, df, taggedDoc=False):
        '''
        Prend en entrée un dataframe df, chaque ligne est un repo avec 'id', 'readme_preproc', 'others_preproc'
        Retourne des listes contenant des documents pour apprendre les 2 modèles Doc2Vec
        '''
        
        #X_readme_others = []
        documents_with_readme = []
        #X_others = []
        documents_with_others = []
        #idx_no_readme = []
        idx_of_repos_without_readme = []
        #id_github = []mlflow_manager
        github_ids = []

        for index, repo in df.iterrows():
            #id_github.append(repo['id'])
            github_ids.append(repo['id'])
            if repo['readme_preproc'] != []:
                #X_readme_others.append(repo['readme_preproc'] + repo['others_preproc'])
                documents_with_readme.append(repo['readme_preproc'] + repo['others_preproc'])
            else:
                #idx_no_readme.append(index)
                idx_of_repos_without_readme.append(index)
            #X_others.append(repo['others_preproc'])
            documents_with_others.append(repo['others_preproc'])

        # Nécessaire pour apprendre le modèle
        # Inutile lors de l'inférence
        if taggedDoc:
            #X_readme_others = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_readme_others)]
            documents_with_readme = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents_with_readme)]
            #X_others = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_others)]
            documents_with_others = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents_with_others)]
    
        #return idx_no_readme, id_github, X_readme_others, X_others
        return idx_of_repos_without_readme, github_ids, documents_with_readme, documents_with_others

    def _get_train_vectors(self, idx_of_repos_without_readme, github_ids):
        '''
        Retourne les vecteurs d'entraînement qui ont été appris durant l'apprentissage sous forme de df
        Trois champs : 'id', '{staging}_readme_vector', '{staging}_other_vector'
        '''
        #vec_readme = []
        readme_vectors = []
        #vec_others = []
        others_vectors = []

        #idx_readme = 0
        idx_of_repos_with_readme = 0

        for ii in range(len(github_ids)):
            #if ii in idx_no_readme:
            if ii in idx_of_repos_without_readme:
                #vec_readme.append(float('nan'))
                #vec_readme.append(np.array([]))
                readme_vectors.append(np.array([]))
            else:
                #vec_readme.append(self._doc2vec_readme.docvecs[idx_readme])
                readme_vectors.append(self._doc2vec_readme.docvecs[idx_of_repos_with_readme])
                #idx_readme += 1
                idx_of_repos_with_readme += 1
            #vec_others.append(self._doc2vec_others.docvecs[ii])
            others_vectors.append(self._doc2vec_others.docvecs[ii])
        
        df_vectors = pd.DataFrame(
            {
                #'id': id_github,
                'id': github_ids,
                #self._type_vec_readme: vec_readme,
                f'{self._staging}_readme_vector': readme_vectors,
                #self._type_vec_others: vec_others
                f'{self._staging}_others_vector': others_vectors,
            }
        )

        return df_vectors

    def _get_test_vectors(self, test_df):
        '''
        Prend un dataframe contenant les répertoires de test en entrée avec 3 champs: 'id', 'readme_preproc', 'others_preproc'
        Retourne un dataframe contenant les répertoires de test avec 3 champs: 'id', '{staging}_readme_vector', '{staging}_others_vector'
        '''
        
        test_df['readme_vectors'] = test_df['readme_preproc'].apply(self._doc2vec_readme.infer_vector)
        test_df['others_vectors'] = test_df['others_preproc'].apply(self._doc2vec_others.infer_vector)

        df = pd.DataFrame(
            {
                'id': test_df['id'],
                f'{self._staging}_readme_vector': test_df['readme_vectors'],
                f'{self._staging}_others_vector': test_df['others_vectors'],
            }
        )
        
        return  df
        
    
    def train(self, train_df, test_df):
        '''
        Entraînement du modèle Doc2Vec from scratch
        train_df --> répertoires d'entraînement
        test_df --> répertoires de test

        retourne les répos de train et de test inférés avec le format suivant:
            'id', '{staging}_readme_vector', '{staging}_others_vector'
        '''
        # Préparation des données
        idx_of_repos_without_readme, github_ids, doc_with_readme, doc_with_others = self._df_to_documents_list(train_df, taggedDoc=True)
        
        # Apprentissage du modèle
        doc2vec_readme = Doc2Vec(doc_with_readme, dm=0, vector_size=300, window=5, dbow_words = 1, min_count=1)
        doc2vec_others = Doc2Vec(doc_with_others, dm=0, vector_size=300, window=5, dbow_words = 1, min_count=1)

        # On stocke les modèles en variable de classe pour les réutiliser dans des méthodes privées plus tard
        self._doc2vec_readme = doc2vec_readme
        self._doc2vec_others = doc2vec_others
        
        # Essayer de définir un path pour log au bon endroit
        #client = MlflowClient()
            
        # Gestion de l'enregistrement du modèle et du versioning
        mlflow_manager = MLFlowManager()
        
        mlflow_manager.register_model(
            model=self._doc2vec_readme,
            model_name='doc2vec_readme',
            artifact_path='../artifacts/doc2vec_readme',
            stage=self._staging,
        )

        mlflow_manager.register_model(
            model=self._doc2vec_others,
            model_name='doc2vec_others',
            artifact_path='../artifacts/doc2vec_others',
            stage=self._staging,
        )
        '''
        with mlflow.start_run() as run:
            # Pickle des modèles pour enregistrer
            # Revoir path des fichiers par la suite
            doc2vec_readme_path = 'doc2vec_readme.pkl'
            doc2vec_others_path = 'doc2vec_others.pkl'

            with open(doc2vec_readme_path, 'wb') as f:
                pickle.dump(doc2vec_readme, f)

            with open(doc2vec_others_path, 'wb') as f:
                pickle.dump(doc2vec_others, f)

            # Enregistrement du modèle appris sur les données avec readme
            mlflow.pyfunc.log_model(
                artifact_path="doc2vec_readme",
                python_model=Doc2VecWraper(),
                artifacts={
                    "model_path": doc2vec_readme_path,
                }
            )

            readme_model_uri = f"runs:/{run.info.run_id}/doc2vec_readme"
            mlflow.register_model(readme_model_uri, "doc2vec_readme")

            # Enregistrement du modèle appris sur les données sans readme
            mlflow.pyfunc.log_model(
                artifact_path="doc2vec_others",
                python_model=Doc2VecWraper(),
                artifacts={
                    "model_path": doc2vec_others_path,
                }
            )

            others_model_uri = f"runs:/{run.info.run_id}/doc2vec_others"
            mlflow.register_model(others_model_uri, "doc2vec_others")

            # Récupération de l'ensemble des versions
            readme_doc2vec_versions = client.search_model_versions(f"name='doc2vec_readme'")
            others_doc2vec_versions = client.search_model_versions(f"name='doc2vec_others'")

            # Récupération de la dernière version enregistrée
            readme_doc2vec_sorted_versions = sorted(readme_doc2vec_versions, key=lambda x: x.creation_timestamp, reverse=True)
            readme_doc2vec_latest_version = readme_doc2vec_sorted_versions[0]

            others_doc2vec_sorted_versions = sorted(others_doc2vec_versions, key=lambda x: x.creation_timestamp, reverse=True)
            others_doc2vec_latest_version = others_doc2vec_sorted_versions[0]

            # Ajout des tags et des alias pour le modèle entraîné sur le readme
            client.set_model_version_tag(
                name="doc2vec_readme",
                version=readme_doc2vec_latest_version.version,
                key="stage",
                value=self._staging,
            )

            client.set_registered_model_alias(
                name="doc2vec_readme",
                alias=self._staging,
                version=readme_doc2vec_latest_version.version
            )

            # Ajout des tags et des alias pour le modèle entraîné sans le readme
            client.set_model_version_tag(
                name="doc2vec_others",
                version=others_doc2vec_latest_version.version,
                key="stage",
                value=self._staging,
            )

            client.set_registered_model_alias(
                name="doc2vec_others",
                alias=self._staging,
                version=others_doc2vec_latest_version.version
            )
        '''

        df_train_vectors = self._get_train_vectors(idx_of_repos_without_readme, github_ids)
        df_test_vectors = self._get_test_vectors(test_df)

        return df_train_vectors, df_test_vectors
        