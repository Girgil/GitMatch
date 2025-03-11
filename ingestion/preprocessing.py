import requests
import pandas as pd
import spacy
import re

from ingestion.requests_utils import check_rate_limit

def load_readme(content_path, token):
    
    url = content_path

    headers = {"Authorization": f"token {token}"}
    
    try:
        try:
            response = requests.get(url, headers=headers)
            check_rate_limit(response)
        except:
            url = url.replace('md', 'rst') # README peut avoir différents suffixe
            response = requests.get(url, headers=headers)
            check_rate_limit(response)
        dictr = response.json()
        url = dictr['download_url']
        response = requests.get(url, headers=headers)
        check_rate_limit(response)
        return response.content.decode("utf-8")
    except:
        return ""

def replace_content_url_by_readme(df: pd.DataFrame, token):
    df2 = df[['contents_url', 'default_branch']]
    df2['contents_url'] = df2.apply(lambda x: load_readme(x['contents_url'][:-7] + f'README.md?ref={x["default_branch"]}', token), axis=1)
    df['contents_url'] = df2['contents_url']
    return df.rename(columns={'contents_url': 'readme'})

def preprocess_repository(df: pd.DataFrame):
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
         'owner_id': df['owner_id'],
        })

def preprocess_df(token, repos, features):
    '''
    Applique le prétraitement aux repos
    '''

    df = pd.DataFrame(repos, columns=features)
    
    df = replace_content_url_by_readme(df, token)
    
    df = preprocess_repository(df)
    
    return df