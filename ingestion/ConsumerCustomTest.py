import json

from pymongo.collection import Collection
from confluent_kafka import Consumer
from utils import save_to_np_array
from preprocessing import preprocess_df
from db_utils import update_users

class ConsumerCustomTest():

    def __init__(self, rep_fullname_already_registered, features=['full_name', 'id', 'description', 'language', 'topics', 'contents_url', 'html_url', 'default_branch', 'owner_id']):
        '''
        Init une liste à vide
        '''
        self._users_contributed_repos = list()
        self._rep_fullname_already_registered = rep_fullname_already_registered
        self._features = features
    
    def _clear(self):
        '''
        Vide la liste
        '''
        self._users_contributed_repos = list()

    def _get_unique_repos_of_multiple_users(self):
        res = []

        for user in self._users_contributed_repos:
            repos = list(user.values())[0]

            for repo in repos:
                if repo['full_name'] not in self._rep_fullname_already_registered:
                    res.append(repo)
                    self._rep_fullname_already_registered.append(repo['full_name'])

        return res

    def _get_repos_id_foreach_user_id(self):
        '''
        retourne un dict avec user_id: [liste de rep_id]
        la variable retournée est adaptée à un envoie en bd
        '''
        res = {}
        
        for user in self._users_contributed_repos:
            user_id = list(user.keys())[0]
            repos = list(user.values())[0]
            
            # Cas où le user possède des repos mais n'y a pas contribué/n'a contribué à aucun repo
            if repos != []:
                repos_id = [repo.get('id') for repo in repos]

                # Format équivalent au groupby nécessaire pour la fonction update_users
                res[user_id] = repos_id
        
        return res

    def consume(self, token, feature_collection: Collection, user_collection: Collection, config, topic, file_to_save_rep_fullname):
        config["group.id"] = "python-group-1"
        config["auto.offset.reset"] = "earliest"
        
        # Instance du consumer
        consumer = Consumer(config)

        # On récupère un topic spécifique
        consumer.subscribe([topic])
        
        try:
            while True:
                # Timeout de 1 seconde
                msg = consumer.poll(1.0)
                if msg is not None and msg.error() is None:
                    # utilité ?
                    print(msg)
                    user_contributed_repos = msg.value()
                    
                    #Accumulation dans la liste
                    self._users_contributed_repos.append(json.loads(user_contributed_repos))

                    # Si dataframe atteint un seuil, il traite et envoie les données en bd
                    if len(self._users_contributed_repos) >= 4:
                        print("On traite 4 repos")
                        # Récupération des user_id de test associée à l'id des repos auxquels ils ont contribué
                        repos_id_foreach_user = self._get_repos_id_foreach_user_id()
                        update_users(repos_id_foreach_user, user_collection, partition='test')
                        
                        # Récupération des repos
                        repos = self._get_unique_repos_of_multiple_users()

                        # Prétraitement
                        df = preprocess_df(token, repos, self._features)
                        
                        # On filtre les colonnes pour ne garder que celles nécessaires pour la bd
                        df_for_features = df.drop(columns=['owner_id'])

                        # Enregistrement en bd des repos
                        df_to_db_features = df_for_features.to_dict(orient="records")
                        feature_collection.insert_many(df_to_db_features)
                        
                        # Nettoyage de la liste
                        self._clear()

                        # On enregistre les nouveaux repos qu'on a envoyé en bd
                        save_to_np_array(file_to_save_rep_fullname, self._rep_fullname_already_registered)
                    
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()