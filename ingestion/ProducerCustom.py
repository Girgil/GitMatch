import time
import json

from datetime import datetime, timedelta
from confluent_kafka import Producer
from ingestion.utils import save_to_np_array
from ingestion.requests_utils import search_repos, filter_repos, filter_features

class ProducerCustom():

    def __init__(self, curr_date, name_repos_registered, features = ['full_name', 'id', 'description', 'language', 'topics', 'contents_url', 'html_url', 'default_branch']):
        '''
        Liste de repos vide
        '''
        self._repos = list()
        self._curr_date = curr_date
        self._name_repos_registered = name_repos_registered
        self._features = features

    def _clear(self):
        self._repos = list()

    def _update_date(self):
        '''
        Méthode qui incrémente la date courante
        '''
    
        # Format de la date
        format_str = "%Y-%m-%d"
    
        # On transforme notre date au format string à un objet datetime
        curr_date_obj = datetime.strptime(self._curr_date, format_str)
    
        # On incrémente d'un jour pour la première requête
        curr_date_obj = curr_date_obj + timedelta(days=1)
    
        # On retransforme nos dates au format string
        self._curr_date = curr_date_obj.strftime(format_str)
        print(f"Nouvelle date : {self._curr_date}")

    def _get_repos(self, token, criteria):
        '''
        Méthode qui requête sur l'api github et qui récupère des repos créés selon une journée
        '''
        headers = {"Authorization": f"token {token}"}
        base_url = "https://api.github.com"
    
        # Url de la requête
        url = f"{base_url}/search/repositories"
        
        '''
        2 types de filtrages en fonction de criteria:
        - modifié le jour curr_date
        - créé le jour curr_date
        '''
        
        params = {
            "q": f"{criteria}:{self._curr_date}",
            "sort": criteria,
            "order": "asc",
            "per_page": 100,
        }

        response = search_repos(url, headers, params)
        
        self._repos = response.json()["items"]
        
        if self._repos !=  []:
            new_repos, self._name_repos_registered = filter_repos(self._repos, self._name_repos_registered)
            
            # On met à jour la date
            self._update_date()

            # On filtre
            self._repos = filter_features(new_repos, self._features)

    def _wait_next_day(self):
        now = datetime.now()
        
        next_day_9am = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        time_to_sleep = (next_day_9am - now).total_seconds()

        print(f"Sleep : {time_to_sleep} seconds")
        time.sleep(time_to_sleep)
    
    def produce(self, topic, config, token, max_date, file_to_save_repos_name, criteria):
        '''
        criteria: created ou pushed
        '''
        # Instance du producer
        producer = Producer(config)
        
        # Condition d'arrêt dépend du type de récolte "pushed" / "created"
        while (True if (criteria == "pushed") else self._curr_date <= max_date):
            # Récupération des repos
            self._get_repos(token, criteria)
            
            # Key de la partition
            key = "0"
            for repo in self._repos:
                repo_json = json.dumps(repo, ensure_ascii=True)
                
                producer.produce(topic, key=key, value=repo_json)
                producer.flush()
    
            self._clear()
            
            # On sauvegarde à chaque itération pour ne pas l'information si le producer s'arrête de manière inattendue
            save_to_np_array(file_to_save_repos_name, self._name_repos_registered)

            # Si le critère est "pushed" on effectue une requête par jour
            # Mise en attente jusqu'à 9 heures le lendemain matin
            if criteria == "pushed":
                self._wait_next_day()
            