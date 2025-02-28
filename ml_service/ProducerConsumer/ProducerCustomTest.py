import json

from confluent_kafka import Producer
from utils import save_to_np_array
from requests_utils import find_repos_with_stars, get_owners, get_repos_for_users, filter_features

class ProducerCustomTest():

    def __init__(self, id_users_registered, features=['full_name', 'id', 'description', 'language', 'topics', 'contents_url', 'html_url', 'default_branch']):
        '''
        Liste de repos vide
        '''
        # Liste de liste
        # Les sous listes contiennent les repos pour lequel un user a contribué
        self._users_contributed_repos = list()
        self._id_users_registered = set(id_users_registered)
        self._features = features

    def _clear(self):
        self._users_contributed_repos = list()

    def _filter_users(self, users):
        '''
        Trie les users dont on a retenu les repos
        Si on l'a déjà enregistré on ne le retient pas
        '''
        users_never_registered = []
        
        for (user_login, user_id) in users:
            if user_id not in self._id_users_registered:
                self._id_users_registered.add(user_id)
                users_never_registered.append((user_login, user_id))
            
        return users_never_registered
        
    def _get_users_contributed_repos(self, token, stars):
        '''
        Récupère les repos d'un set de users
        '''
        # Récupère des repos en fonction d'un nombre précis de stars
        original_repos = find_repos_with_stars(stars=stars, token=token)

        # Récupère les owners de ces repos
        users = get_owners(original_repos)

        # Filtre les users dont on a déjà enregistré les repos
        users = self._filter_users(users)

        self._users_contributed_repos = get_repos_for_users(token, users)

        # On filtre la plupart des informations du repo pour en transmettre seulement certaines
        for ii, user_contributed_repos in enumerate(self._users_contributed_repos):
            repos = list(user_contributed_repos.values())[0]
            self._users_contributed_repos[ii] = {list(user_contributed_repos.keys())[0]: filter_features(repos, self._features)}
    
    def produce(self, topic, config, token, min_stars, max_stars, file_to_save_users_id_registered):
        # Instance du producer
        producer = Producer(config)

        curr_stars = min_stars

        while curr_stars < max_stars:
            # Récupération des repos pour chaque user de test
            self._get_users_contributed_repos(token, curr_stars)
            
            curr_stars += 1

            key = "0"
            for user_contributed_repos in self._user_contributed_repos:
                user_contributed_repos_json = json.dumps(user_contributed_repos, ensure_ascii=True)

                producer.produce(topic, key=key, value=user_contributed_repos_json)
                producer.flush()

            self._clear()

            # On sauvegarde les users dont on a déjà enregistré les repos
            save_to_np_array(file_to_save_users_id_registered, self._id_users_registered)