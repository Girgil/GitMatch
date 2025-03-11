import pandas as pd
import random

class DatabaseManager():

    def __init__(self, database):
        """
        Initialise un objet DatabaseManager qui sert d'interface
        entre une database mongo et des programmes tiers
        """
        self._database = database

    def insert_repos_features(self, df_to_insert):
        """
        Insère des répertoires dans la collection des features
        Prend en entrée un pd.DataFrame à 4 colonnes:
            - id: identifiant du répertoire
            - readme_preproc: tokens du répertoire
            - others_preproc: tokens des autres features
            - html_url: url du répertoire
        """

        features_collection = self._database['repertoire_features']

        oriented_df_to_insert = df_to_insert.to_dict(orient="records")

        features_collection.insert_many(oriented_df_to_insert)

    def insert_users(self, users_contributed_repos, partition):
        """
        Insère des utilisateurs dans la collection des users
        Users_contributed_repos est un dict qui associe l'identifiant d'un utilisateur
        aux identifiants des répertoires auxquels il a contribué
        Partition indique si les utilisateurs seront utilisés en 'train' ou en 'test'
        """
        
        users_collection = self._database['users']

        for user_id, repos_id in users_contributed_repos.items():
            criteria = {'user_id': user_id}

            users_collection.update_one(
                criteria,
                {
                    '$setOnInsert': {'partition': partition},
                    '$addToSet': {'rep_id': {'$each': repos_id}}
                },
                upsert=True
            )

    def insert_repos_vectors_for_one_stage(self, df_to_insert, staging):
        """
        Insère les représentations vectorielles de répertoires pour un type de modèle
        'staging' ou 'production'
        """

        vectors_collection = self._database['repertoire_vectors']

        stage_rd_vector = f'{staging}_readme_vector'
        stage_ot_vector = f'{staging}_others_vector'

        oriented_df_to_insert = df_to_insert.to_dict(orient="records")

        for repo in oriented_df_to_insert:
            criteria = {'id': repo['id']}

            vectors_collection.update_one(
                criteria,
                {
                    '$set': {
                        stage_rd_vector: repo[stage_rd_vector].tolist(),
                        stage_ot_vector: repo[stage_ot_vector].tolist(),
                    }
                },
                upsert=True
            )

    def insert_repos_vector_for_both_stages(self, df_to_insert):
        """
        Insère les représentations vectorielles de répertoires pour les deux types
        de modèle
        """

        vectors_collection = self._database['repertoire_vectors']

        stg_rd_vector = 'staging_readme_vector'
        stg_ot_vector = 'staging_others_vector'
        prd_rd_vector = 'production_readme_vector'
        prd_ot_vector = 'production_others_vector'

        oriented_df_to_insert = df_to_insert.to_dict(orient="records")

        for repo in oriented_df_to_insert:
            criteria = {'id': repo['id']}

            vectors_collection.update_one(
                criteria,
                {
                    '$set': {
                        stg_rd_vector: repo[stg_rd_vector].tolist(),
                        stg_ot_vector: repo[stg_ot_vector].tolist(),
                        prd_rd_vector: repo[prd_rd_vector].tolist(),
                        prd_ot_vector: repo[prd_ot_vector].tolist(),
                    }
                },
                upsert=True
            )

    def get_url_list_from_id_list(self, ids_list):
        """"
        Retourne l'url des répertoires dont les identifiants sont dans la liste
        """
        features_collection = self._database['repertoire_features']

        urls_list = list(
            features_collection.find(
                {'id': {'$in': ids_list}},
                {
                    'html_url': 1,
                }
            )
        )

        # Garder seulement la valeur html_url
        urls_list = [url['html_url'] for url in urls_list]

        return urls_list

    def get_train_test_split_features(self):
        """
        Retourne deux pd.DataFrame: un pour le train; un pour le test
        Composés de 3 champs: 'id', 'readme_preproc', 'others_preproc'
        """
        features_collection = self._database['repertoire_features']
        users_collection = self._database['users']

        # Récupérer l'id de tous les répertoires utilisés pour le test
        test_repos_id_for_each_test_user = list(
            users_collection.find(
                {'partition': 'test'},
                {'rep_id': 1},
            )
        )
        # Garder les répos uniques
        unique_test_repos_id = set()

        for test_repos in test_repos_id_for_each_test_user:
            unique_test_repos_id.update(test_repos['rep_id'])

        unique_test_repos_id = list(unique_test_repos_id)

        # Récupérer les répertoires de test et en faire un df
        test_repos = list(
            features_collection.find(
                {'id': {'$in': unique_test_repos_id}},
                {
                    'id': 1,
                    'readme_preproc': 1,
                    'others_preproc': 1,
                },
            )
        )

        df_test_repos = pd.DataFrame(
            test_repos,
            columns=[
                'id',
                'readme_preproc',
                'others_preproc',
            ]
        )

        # Récupérer les répertoires de train et en faire un df
        train_repos = list(
            features_collection.find(
                {'id': {'$nin': unique_test_repos_id}},
                {
                    'id': 1,
                    'readme_preproc': 1,
                    'others_preproc': 1,
                },
            )
        )

        df_train_repos = pd.DataFrame(
            train_repos,
            columns=[
                'id',
                'readme_preproc',
                'others_preproc',
            ]
        )

        return df_train_repos, df_test_repos



    def get_df_vectors(self):
        """
        Récupère pour chaque répertoire l'ensemble de ses représentations vectorielles
        Retourne un pd.DataFrame
        """

        vectors_collection = self._database['repertoire_vectors']

        #vectors_list = list(vectors_collection.find())
        vectors_list = list(
            vectors_collection.find(
                {},
                {
                    'id': 1,
                    'staging_readme_vector': 1,
                    'staging_others_vector': 1,
                    'production_readme_vector': 1,
                    'production_others_vector': 1,
                }
            )
        )

        cols = [
            'id',
            'staging_readme_vector',
            'staging_others_vector',
            'production_readme_vector',
            'production_others_vector',
        ]
        
        df_vectors = pd.DataFrame(
            vectors_list,
            columns=cols
        )

        return df_vectors

    def get_df_split_staging_production(self, df):
        """
        Sépare un DataFrame en deux selon l'origine de la représentation vectorielle:
        'staging' ou 'production'
        """

        stg_cols = [
            'id',
            'staging_readme_vector',
            'staging_others_vector',
        ]

        prd_cols = [
            'id',
            'production_readme_vector',
            'production_others_vector',
        ]
        
        staging_df = df[stg_cols]
        production_df = df[prd_cols]

        return staging_df, production_df

    def get_df_split_train_test_vectors(self):
        """
        Retourne deux pd.DataFrame avec les vecteurs de train et de test séparés
        """

        users_collection = self._database['users']

        df_vectors = self.get_df_vectors()

        # Récupérer l'id de tous les répertoires utilisés pour le test
        test_repos_id_for_each_test_user = list(
            users_collection.find(
                {'partition': 'test'},
                {'rep_id': 1},
            )
        )
        # Garder les répos uniques
        unique_test_repos_id = set()

        for test_repos in test_repos_id_for_each_test_user:
            unique_test_repos_id.update(test_repos['rep_id'])

        unique_test_repos_id = list(unique_test_repos_id)

        #train_df = df_vectors[df_vectors['id'] not in unique_test_repos_id]
        train_df = df_vectors[~df_vectors['id'].isin(unique_test_repos_id)]
        #test_df = df_vectors[df_vectors['id'] in unique_test_repos_id]
        test_df = df_vectors[df_vectors['id'].isin(unique_test_repos_id)]

        return train_df, test_df

    def get_users_test_vectors_splitted(self):
        """
        Prépare les données des utilisateurs de test pour l'évaluation
        Tire aléatoirement pour chaque utilisateur de test un répertoire à mettre de côté
        Il servira pour récupérer les k répertoires les plus similaires afin d'évaluer les modèles
        """

        users_collection = self._database['users']
        
        _, test_df = self.get_df_split_train_test_vectors()

        users = list(
            users_collection.find(
                {'partition': 'test'},
                {'rep_id': 1},
            )
        )

        users_rep_ids = []
        
        for user in users:
            # user['rep_id']
            users_rep_ids.append(user['rep_id'])

        # On va tirer aléatoirement un répertoire de test pour chaque utilisateur de test
        users_random_chosen_vectors = []
        users_remaining_test_vectors = []

        for ii, rep_ids in enumerate(users_rep_ids):
            random_index = random.randint(0, len(rep_ids)-1)

            users_random_chosen_vectors.append(rep_ids[random_index])
            del users_rep_ids[ii][random_index]

            # Transformation des id en dict/df
            #users_random_chosen_vectors[ii] = test_df[test_df['id'] == users_random_chosen_vectors[ii]].to_dict(orient="records")
            users_random_chosen_vectors[ii] = (
                test_df[test_df['id'] == users_random_chosen_vectors[ii]]
                .to_dict(orient="records")[0]
            )
            
            users_remaining_test_vectors.append(
                pd.DataFrame(
                    [
                        test_df[test_df['id'] == rep_id].to_dict(orient="records")[0]
                        for rep_id in users_rep_ids[ii]
                    ],
                    columns=[
                        'id',
                        'staging_readme_vector',
                        'staging_others_vector',
                        'production_readme_vector',
                        'production_others_vector',
                    ]
                )
            )

        return users_random_chosen_vectors, users_remaining_test_vectors

    
    def upgrade_staging_vectors(self):
        """
        Transfère les représentations vectorielles de staging en production
        """
        vectors_collection = self._database['repertoire_vectors']
        
        # Récupération du dataframe de s
        df_vectors = self.get_df_vectors()

        stg_df, _ = self.get_df_split_staging_production(df_vectors)

        stg_vectors_to_production = stg_df.to_dict(orient='records')

        for staging_vectors in stg_vectors_to_production:
            vectors_collection.update_one(
                {'id': staging_vectors['id']},
                {
                    '$set': {
                        'production_readme_vector': staging_vectors['staging_readme_vector'],
                        'production_others_vector': staging_vectors['staging_others_vector']
                    }
                },
                upsert=True
            )

    def get_repos_without_all_vectors(self):
        """
        Retourne les répertoires qui n'ont pas leurs quatres représentations vectorielles enregistrées
        en collection
        """
        features_collection = self._database['repertoire_features']
        vectors_collection = self._database['repertoire_vectors']
        
        # Récupérer tous les id des répertoires qui ont des features dans repertoire_features
        repos = list(
            features_collection.find(
                {},
                {
                    'id': 1,
                }
            )
        )

        # Filtrage pour passer d'une liste de dict à une liste d'id
        repos_id = [repo['id'] for repo in repos]
        
        # Garder tous les ids des répertoires qui n'ont pas 4 représentations vectorielles dans repertoire_vectors
        repos = list(
            vectors_collection.find(
                {
                    'id': {'$in': repos_id},
                    'staging_readme_vector': {'$ne': None},
                    'staging_others_vector': {'$ne': None},
                    'production_readme_vector': {'$ne': None},
                    'production_others_vector': {'$ne': None}
                },
                {
                    'id': 1,
                }
            )
        )

        repos_id_with_all_vectors = [repo['id'] for repo in repos]

        # On récupère tous les répertoires qui n'ont pas leur quatre représentations
        repos_id_to_predict = list(set(repos_id) - set(repos_id_with_all_vectors))
        
        # Récupérer les repos dont il faut obtenir une représentation
        repos_to_predict = list(
            features_collection.find(
                {'id': {'$in': repos_id_to_predict}},
                {
                    'id': 1,
                    'readme_preproc': 1,
                    'others_preproc': 1,
                }
            )
        )

        # Retourner un DataFrame

        df_to_predict = pd.DataFrame(
            repos_to_predict,
            columns=[
                'id',
                'readme_preproc',
                'others_preproc',
            ]
        )

        return df_to_predict

        