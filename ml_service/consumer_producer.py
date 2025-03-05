import os
import re
from datetime import datetime, timedelta
from os import path

import numpy as np
import pandas as pd
import requests
import spacy
from dotenv import load_dotenv
from pymongo import MongoClient


def load_info_repos():
    '''
    Méthode qui charge des informations relatives aux dépos déjà enregistrés: 
        - date de création du dernier repo enregistré 
        - nom des repos déjà enregistrés
    '''

    ''' 
    Date au format ISO 8601 : Année-Mois-Jour
    On commence à les récupérer début juillet 2010
    En juillet 2010, GitHub annonçait avoir 1 million de répertoires en ligne
    '''

    '''
    Données à l'initialisation
    '''
    start_date = '2010-07-01'
    name_repos = np.array([])

    path_date_save = 'save_date_file.txt'
    path_name_repos_save = 'name_repos_saved.npy'
    
    # On charge les infos
    if path.isfile(path_date_save):
        with open(path_date_save, "r") as date_f:
            start_date = next(date_f).replace("\n", "")

    if path.isfile(path_name_repos_save):
        name_repos = np.load(path_name_repos_save)

    return start_date, name_repos


def save_infos_repos(start_date, name_repos):
    '''
    Méthode qui sauvegarde des informations relatives aux dépos enregistrés:
        - date de création du dernier repo enregistré
        - nom des repos déjà enregistrés
    '''

    file_date_save = 'save_date_file.txt'
    file_name_repos_save = 'name_repos_saved.npy'

    # Format de la date
    format_str = "%Y-%m-%d"

    # On transforme notre date au format string à un objet datetime
    start_date_obj = datetime.strptime(start_date, format_str)

    # On incrémente d'un jour pour la première requête
    start_date_obj = start_date_obj + timedelta(days=1)

    # On retransforme nos dates au format string
    start_date = start_date_obj.strftime(format_str)

    with open(file_date_save, "w") as date_f:
        date_f.write(f"{start_date}\n")

    with open(file_name_repos_save, 'wb') as name_repos_f:
        np.save(name_repos_f, name_repos)


def filter_repos(new_repos, name_repos_saved):
    '''
    Méthode qui fitre les repos déjà enregistrés à partir de leur nom
    '''

    '''
    On transforme name_repos_saved en set
    Ça facilite la recherche des répos déjà présents dans ceux enregstrés
    '''
    name_repos_saved = set(name_repos_saved)
    new_repos_to_save = []
    name_new_repos_to_save = set()

    for repo in new_repos:
        if repo['full_name'] not in name_repos_saved:
            new_repos_to_save.append(repo)
            name_new_repos_to_save.add(repo['full_name'])

    new_repos_to_save = np.array(new_repos_to_save)
    name_repos_to_save = np.array(list(name_repos_saved.union(name_new_repos_to_save)))
    
    return new_repos_to_save, name_repos_to_save


def get_repos(token):
    '''
    Méthode qui requête sur l'api github et qui récupère ...
    '''
    
    # Chargement de la config
    headers = {"Authorization": f"token {token}"}
    base_url = "https://api.github.com"

    # Url de la requête
    url = f"{base_url}/search/repositories"
    
    
    # Nouveau répertoire
    new_repos = []

    # Chargement des informations de la dernière récupération
    start_date, name_repos_saved = load_info_repos()

    '''
    Paramètres de la requête:
        - pas de filtrage des dépôts sur des informations caractéristiques
        - filtrage à partir de la date de création; on ne récupère pas de dépôts plus vieux qu'une date limite
    '''
    
    params = {
        "q": f"created:{start_date}",
        "sort": "created",
        "order": "asc",
        "per_page": 100,
    }

    # On fait la requête
    response = requests.get(url, headers=headers, params=params)


    if response.status_code == 200:
        repos = response.json()["items"]
        if repos:
            new_repos = repos
        else:
            print(f"Aucun dépôt trouvé créé {start_date}.")
            return None
    else:
        print(f"Erreur lors de la recherche du dépôt : {response.status_code} - {response.text}")
        return None

    '''
    Features du répertoire que l'on garde pour les prétraitements du consumer
    Le full name n'est utile que pour le filtrage des repos déjà sauvegardés
    '''
    features = ['full_name', 'id', 'description', 'language', 'topics', 'contents_url', 'html_url', 'default_branch']

    
    # On filtre les repos récupérés pour qu'ils ne soient pas enregistrés deux fois
    new_repos, name_repos_to_save = filter_repos(repos, name_repos_saved)
    
    # On sauvegarde les infos des repos
    save_infos_repos(start_date, name_repos_to_save)

    repos_df = pd.DataFrame.from_dict(repos)[features]
    
    return repos_df

def load_readme(content_path, token):
    
    url = content_path

    headers = {"Authorization": f"token {token}"}
    
    try:
        try:
            response = requests.get(url, headers=headers)
        except:
            url = url.replace('md', 'rst') # README peut avoir différents suffixe
            response = requests.get(url, headers=headers)
        dictr = response.json()
        url = dictr['download_url']
        response = requests.get(url, headers=headers)
        return response.content.decode("utf-8")
    except:
        return ""


def replace_content_url_by_readme(df, token):
    df2 = df[['contents_url', 'default_branch']]
    df2['contents_url'] = df2.apply(lambda x: load_readme(x['contents_url'][:-7] + f'README.md?ref={x["default_branch"]}', token), axis=1) 
    df['contents_url'] = df2['contents_url']
    return df.rename(columns={'contents_url': 'readme'})


def preproc_rep(df):
    nlp = spacy.load("en_core_web_lg")

    readme_preproc = []
    doc_preproc = []
    ids_preproc = []

    for index, row in df.iterrows():

        doc = ''
        ids_preproc.append(row['id'])
        topics = ''
        for ii in row['topics']:
            topics += re.sub(r"[^\w\s]", " ", ii ) + ' '
        
        if type(row['description']) == float or type(row['description']) == type(None):
            description = ''
        else:
            description = re.sub(r"[^\w\s]", " ", row['description'] )
            
        if type(row['language']) == float or type(row['language']) == type(None):
            language = ''
        else:
            language = re.sub(r"[^\w\s]", " ", row['language'] )
        
        doc += description + ' ' + language + ' ' + topics
    
        if type(row['readme']) == float or type(row['readme']) == type(None):
            readme_preproc.append([])
        else:
            readme = re.sub(r"[^\w\s]", " ", row['readme'] )
            doc_readme = readme
            tokenized = [token.text.lower() for token in nlp(doc_readme) if not token.is_punct and not token.is_space and not token.like_url and len(token.text) > 2 and len(token.text) <= 20]
            readme_preproc.append(tokenized)
    
        tokenized = [token.text.lower() for token in nlp(doc) if not token.is_punct and not token.is_space and not token.like_url and len(token.text) > 2 and len(token.text) <= 20]
        doc_preproc.append(tokenized)


    return pd.DataFrame(
        {'id': ids_preproc,
         'readme_preproc': readme_preproc,
         'others_preproc': doc_preproc,
         'html_url': df['html_url'],
        })


if __name__ == "__main__":
    spacy.cli.download("en_core_web_lg")

    # Load variabels d'environnement
    env_path = '../.env'

    load_dotenv(env_path)

    token = os.getenv("TOKEN_GITHUB")
    mongo_uri = os.getenv("MONGO_URI")

    # Load dbmongo
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client["github"]
    collection = db["repos"]

    # Charger des fichiers
    df_repos = get_repos(token)

    # Prétraitement
    df_repos_with_readme_loaded = replace_content_url_by_readme(df_repos)

    df_repos_preprocessed = preproc_rep(df_repos_with_readme_loaded)

    # Enregistrement en bd
    df_to_db = df_repos_preprocessed.to_dict(orient="records")

    collection.insert_many(df_to_db)
    