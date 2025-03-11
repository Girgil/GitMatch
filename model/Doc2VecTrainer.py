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
        """"
        Prend en entrée un dataframe df, chaque ligne est un repo avec 'id', 'readme_preproc', 'others_preproc'
        Retourne des listes contenant des documents pour apprendre les 2 modèles Doc2Vec
        """
        
        documents_with_readme = []
        documents_with_others = []
        
        idx_of_repos_without_readme = []
        github_ids = []

        for index, repo in df.iterrows():
            github_ids.append(repo['id'])
            if repo['readme_preproc'] != []:
                documents_with_readme.append(repo['readme_preproc'] + repo['others_preproc'])
            else:
                idx_of_repos_without_readme.append(index)
            documents_with_others.append(repo['others_preproc'])

        # Nécessaire pour apprendre le modèle
        # Inutile lors de l'inférence
        if taggedDoc:
            documents_with_readme = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents_with_readme)]
            documents_with_others = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents_with_others)]
    
        return idx_of_repos_without_readme, github_ids, documents_with_readme, documents_with_others

    def _get_train_vectors(self, idx_of_repos_without_readme, github_ids):
        """
        Retourne les vecteurs d'entraînement qui ont été appris durant l'apprentissage sous forme de df
        Trois champs : 'id', '{staging}_readme_vector', '{staging}_other_vector'
        """
        readme_vectors = []
        others_vectors = []

        idx_of_repos_with_readme = 0

        for ii in range(len(github_ids)):
            if ii in idx_of_repos_without_readme:
                readme_vectors.append(np.array([]))
            else:
                readme_vectors.append(self._doc2vec_readme.docvecs[idx_of_repos_with_readme])
                idx_of_repos_with_readme += 1
            others_vectors.append(self._doc2vec_others.docvecs[ii])
        
        df_vectors = pd.DataFrame(
            {
                'id': github_ids,
                f'{self._staging}_readme_vector': readme_vectors,
                f'{self._staging}_others_vector': others_vectors,
            }
        )

        return df_vectors

    def _get_test_vectors(self, test_df):
        """
        Prend un dataframe contenant les répertoires de test en entrée avec 3 champs: 'id', 'readme_preproc', 'others_preproc'
        Retourne un dataframe contenant les répertoires de test avec 3 champs: 'id', '{staging}_readme_vector', '{staging}_others_vector'
        """
        
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
        """
        Entraînement du modèle Doc2Vec from scratch
        train_df --> répertoires d'entraînement
        test_df --> répertoires de test

        retourne les répos de train et de test inférés avec le format suivant:
            'id', '{staging}_readme_vector', '{staging}_others_vector'
        """
        # Préparation des données
        idx_of_repos_without_readme, github_ids, doc_with_readme, doc_with_others = self._df_to_documents_list(train_df, taggedDoc=True)
        
        # Apprentissage du modèle
        doc2vec_readme = Doc2Vec(doc_with_readme, dm=0, vector_size=300, window=5, dbow_words = 1, min_count=1)
        doc2vec_others = Doc2Vec(doc_with_others, dm=0, vector_size=300, window=5, dbow_words = 1, min_count=1)

        # On stocke les modèles en variable de classe pour les réutiliser dans des méthodes privées plus tard
        self._doc2vec_readme = doc2vec_readme
        self._doc2vec_others = doc2vec_others
            
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

        df_train_vectors = self._get_train_vectors(idx_of_repos_without_readme, github_ids)
        df_test_vectors = self._get_test_vectors(test_df)

        return df_train_vectors, df_test_vectors
        