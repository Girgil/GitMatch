import json

from pymongo.collection import Collection
from confluent_kafka import Consumer
from preprocessing import preprocess_df
from db_utils import update_users

class ConsumerCustom():

    def __init__(self, features=['full_name', 'id', 'description', 'language', 'topics', 'contents_url', 'html_url', 'default_branch', 'owner_id']):
        '''
        '''
        self._repos = list()
        self._features = features
    
    def _clear(self):
        '''
        Vide la liste
        '''
        self._repos = list()

    def consume(self, token, feature_collection: Collection, user_collection: Collection, config, topic):
        '''
        Boucle de récupération 
        '''
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
                    print(msg)
                    repo_json = msg.value()
                    
                    #Accumulation dans le dataframe
                    self._repos.append(json.loads(repo_json))

                    # Si dataframe atteint un seuil, il traite et envoie les données en bd
                    if len(self._repos) >= 50:
                        print("On traite 50 repos")
                        # Prétraitement des repos
                        df = preprocess_df(token, self._repos, self._features)

                        # On sépare les champs du df retourné en plusieurs sous df
                        df_for_features = df.drop(columns=['owner_id'])
                        df_for_users = df.drop(columns=['readme_preproc', 'others_preproc', 'html_url'])
                        
                        # Enregistrement en bd des repos
                        df_to_db_features = df_for_features.to_dict(orient="records")
                        feature_collection.insert_many(df_to_db_features)
                        
                        # Enregistrement en bd du lien entre user et contribution
                        owner_id_to_repos_id = df_for_users.groupby('owner_id')['id'].apply(list).to_dict()
                        update_users(owner_id_to_repos_id, user_collection, partition='train')

                        # Nettoyage de la liste
                        self._clear()
                    
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()