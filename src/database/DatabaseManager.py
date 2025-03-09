import pandas as pd
import random

class DatabaseManager():

    def __init__(self, database):
        '''
        database: database mongo
        '''
        self._database = database

    def insert_repos_features(self, df_to_insert):
        '''
        df_to_insert: dataframe à 4 colonnes 'id', 'readme_preproc', 'others_preproc', 'url'
        '''

        features_collection = self._database['repertoire_features']

        oriented_df_to_insert = df_to_insert.to_dict(orient="records")

        features_collection.insert_many(oriented_df_to_insert)

    def insert_users(self, users_contributed_repos, partition):
        '''
        users_contributed_repos : dict qui associe un user_id aux repos possédés/contribués
        
        partition : test ou train, indique pour quelle partition l'utilisateur est utilisé
        
        Met à jour en bd la liste des répertoires auxquels un utilisateur a contribué
        '''
        
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
        '''
        df_to_insert: df avec 3 champs id "{staging}_readme_vector", "{staging}_others_vector"
        Insère des vecteurs pour un stage "staging"/"production"
        '''

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
        '''
        df_to_insert: df avec 5 champs id et les 4 vecteurs
        Insère des vecteurs pour les deux stages "staging" et "production"
        '''

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
        '''
        Retourne l'url des repos dont les id sont dans la liste
        '''
        features_collection = self._database['repertoire_features']

        urls_list = list(
            features_collection.find(
                {'id': {'$in': ids_list}},
                {
                    'html_url': 1,
                }
            )
        )

        return urls_list

    def get_df_staging_production_split(self, df):
        '''
        prend en entrée un df à 5 colonnes: 'id' + 4 vecteurs et retourne deux df pour le staging et la production
        '''
        

    def get_df_vectors(self):
        '''
        Retourne deux dataframes contenant l'id (nécessité ?), un champ de vecteur pour et pour others
        --> un dataframe par stage (staging/production)
        '''

    def get_random_users_split_repos(self):
        '''
        Retourne 2 listes
        1 liste qui contient un repo tiré aléatoirement pour chaque utilisateur --> liste de dict à 4 champs / 5 champs (id?)
        1 liste qui contient des df à 5 champs (id?), chaque dataframe correspond aux répertoires contribués par l'utilisateur moins celui tiré pour le test 
        '''

    def get_train_test_split_features(self):
        '''
        Retourne deux dataframes un pour le train / un pour le test avec les features (readme_preproc/others_preproc) et l'id
        '''
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
        '''
        M2
        retourne tous les vecteurs sous forme d'un df
        '''

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
        '''
        M1
        split un df en deux pour séparer staging et production
        '''

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
        '''
        M3
        retourne deux df avec les vecteurs de train et de test séparés
        '''

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
        '''
        M4
        retourne deux listes
        une première avec une liste de dict qui contient chaque repo tiré aléatoirement pour les users de test
        une seconde avec une liste de df qui correspond aux répertoires restants des utilisateurs de test
        '''

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
        '''
        Transfère les vecteurs de staging en vecteurs de production
        '''
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
        