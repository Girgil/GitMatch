import requests
import numpy as np
import time

def search_repos(url, headers, params):
    """
    Applique une recherche
    """
    try:
        response = requests.get(url, headers=headers, params=params)
        check_rate_limit(response)
        return response
    except:
        return []

def filter_repos(repos, name_repos_registered):
    """
    Méthode qui fitre les repos déjà enregistrés à partir de leur nom
    """
    set_name_repos_registered = set(name_repos_registered)
    
    repos_to_register = []
    set_name_repos_to_register = set()

    for repo in repos:
        if repo['full_name'] not in set_name_repos_registered:
            repos_to_register.append(repo)
            set_name_repos_to_register.add(repo['full_name'])

    repos_to_register = np.array(repos_to_register)
    
    name_repos_registered = np.array(list(set_name_repos_registered.union(set_name_repos_to_register)))
    
    return repos_to_register, name_repos_registered

def filter_features(repos, features):
    """
    Filtrage des répertoires au format json pour ne garder que certaines features
    """
    return [{feature: repo[feature] for feature in features} | {'owner_id': repo['owner']['id']} for repo in repos]

def find_repos_with_stars(stars, token):
    """
    """
    headers = {"Authorization": f"token {token}"}
    base_url = "https://api.github.com"
    
    url = f"{base_url}/search/repositories"
    ### On précise qu'on veut min_stars étoiles précisément ###
    params = {"q": f"stars:{stars}", "sort": "stars", "order": "desc", "per_page": 5}
    
    response = search_repos(url, headers, params)
    
    return response.json()["items"]

def get_owners(repos):
    """
    Récupération du propriétaire des répertoires
    """
    owners_id = set()

    for repo in repos:
        owners_id.add((repo['owner']['login'], repo['owner']['id']))

    return list(owners_id)

def get_repos_for_users(token, users):
    """
    Récupère tous les répertoires auxquels une liste d'utilisateurs a contribué
    Contributions: Commits, Pull-requests, Issues
    """
    res = []

    for num_user, (username, user_id) in enumerate(users):
        print(f'Curr user: {num_user} - {username} - {user_id}')
        
        repos_from_commits = get_repos_from_commits(token, username)
        repos_from_prs = get_repos_from_pull_requests(token, username)
        repos_from_issues = get_repos_from_issues(token, username)

        unique_repos = get_unique_repos(user_id, 
                                        repos_from_commits, 
                                        repos_from_prs, 
                                        repos_from_issues)
        res.append(unique_repos)

    return res

def get_repos_from_commits(token, username, max_results=100000000):
    """
    Récupère les répertoires depuis les commits d'un utilisateur
    """
    base_url = "https://api.github.com"
    headers = {"Authorization": f"token {token}"}
    url = f"{base_url}/search/commits"

    params = {"q": f"author:{username}", "per_page": 100}

    commits = []
    repos_set = {}

    while url and len(commits) < max_results:
        response = search_repos(url, headers, params)
        new_commits = response.json()["items"]

        commits.extend(new_commits)

        for commit in new_commits:
            repo_url = commit.get('repository').get('url')
            if repo_url:
                repo = get_repo_info(repo_url, headers)
                if repo:
                    repos_set[repo['id']] = repo 

        url = get_next_page_url(response.headers)

    return list(repos_set.values())

def get_repos_from_pull_requests(token, username, max_results=100000):
    """
    Récupère les répertoires depuis les pull requests d'un utilisateur
    """
    base_url = "https://api.github.com"
    headers = {"Authorization": f"token {token}"}
    url = f"{base_url}/search/issues"
    
    params = {"q": f"is:pull-request author:{username}", "per_page": 100}
    
    prs = []
    repos_set = {}
    
    while url and len(prs) < max_results:
        response = search_repos(url, headers, params)
        new_prs = response.json()["items"]
        
        prs.extend(new_prs)

        for pr in new_prs:
            repo_url = pr.get('repository_url')
            if repo_url:
                repo_info = get_repo_info(repo_url, headers)
                if repo_info:
                    repos_set[repo_info['id']] = repo_info

        url = get_next_page_url(response.headers)
            
    return list(repos_set.values())

def get_repos_from_issues(token, username, max_results=100000):
    """
    Récupère les répertoires depuis les issues d'un utilisateur
    """
    base_url = "https://api.github.com"
    headers = {"Authorization": f"token {token}"}
    url = f"{base_url}/search/issues"
    
    params = {"q": f"is:issue author:{username}", "per_page": 100}
    
    issues = []
    repos_set = {}

    while url and len(issues) < max_results:
        response = search_repos(url, headers, params)
        
        new_issues = response.json()["items"]
        
        issues.extend(new_issues)

        for issue in new_issues:
            repo_url = issue.get('repository_url')
            if repo_url:
                repo_info = get_repo_info(repo_url, headers)
                
                if repo_info:
                    repos_set[repo_info['id']] = repo_info

        url = get_next_page_url(response.headers)
            
    return list(repos_set.values())

def get_repo_info(repo_url, headers):
    """
    Récupère les informations d'un répertoire à partir de son url
    """
    response = requests.get(repo_url, headers=headers)
    check_rate_limit(response)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur lors de la récupération des informations du dépôt: {response.status_code}")
        return None

def check_rate_limit(response):
    """
    Vérifie les limites de taux du token d'accès
    Même en attente si la limite est atteinte
    """
    if 'X-RateLimit-Remaining' in response.headers:
        remaining = int(response.headers['X-RateLimit-Remaining'])
        print(f"Requêtes restantes (global): {remaining}")
        if remaining == 0:
            reset_time = int(response.headers['X-RateLimit-Reset'])
            sleep_time = reset_time - int(time.time())
            if sleep_time < 0:
                sleep_time = 0
            print(f"Limite de taux globale atteinte. Attente de {sleep_time} secondes.")
            time.sleep(sleep_time)

def get_unique_repos(user_id, repos_from_commits, repos_from_prs, repos_from_issues):
    """
    Pour un utilisateur filtre les doublons des répertoires
    """
    repos_set = {}

    for repo_from_commit in repos_from_commits:
        repos_set[repo_from_commit.get('id')] = repo_from_commit

    for repo_from_pr in repos_from_prs:
        repos_set[repo_from_pr.get('id')] = repo_from_pr

    for repo_from_issue in repos_from_issues:
        repos_set[repo_from_issue.get('id')] = repo_from_issue

    return {user_id: list(repos_set.values())}

def get_next_page_url(headers):
    """
    Permet de requêter sur la page suivante
    """
    if 'Link' in headers:
        links = headers['Link'].split(', ')
        for link in links:
            if 'rel="next"' in link:
                return link.split(';')[0].strip('<>')
    return None

def get_repo(token, full_name):
    """
    Récupération du répertoire en inférence
    """
    headers = {"Authorization": f"token {token}"}
    base_url = "https://api.github.com"
    
    url = f"{base_url}/repos/{full_name}"
    
    response = search_repos(url, headers, params=None)
    repo = response.json()

    return repo
    