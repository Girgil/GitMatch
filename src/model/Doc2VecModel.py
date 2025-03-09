import mlflow
from model_utils import compute_similarity_cosinus

class Doc2VecModel():

    def __init__(self, staging, mlflow_readme_model_uri, mlflow_others_model_uri):
        self._doc2vec_readme = mlflow.pyfunc.load_model(mlflow_readme_model_uri)
        self._doc2vec_others = mlflow.pyfunc.load_model(mlflow_others_model_uri)

        self._staging = staging

    def predict(self, repo):
        '''
        Prend en entrée un repo représenté par un dict à 3 champs: 'id', 'readme_preproc', 'others_preproc'

        Retourne un dict formaté selon 3 champs: 'id', '{staging}_readme_vector', '{staging}_others_vector'
        '''
        readme_vector = self._doc2vec_readme.predict(repo['readme_preproc'])
        others_vector = self._doc2vec_others.predict(repo['others_preproc'])

        return {'id': repo['id'], f'{self._staging}_readme_vector': readme_vector, f'{self._staging}_others_vector': others_vector}

    def _get_top_k(self, k, readme_vector, others_vector, df_vectors):
        '''
        k nombre de répertoires qu'on retourne 'id', '', 'readme_similarity_value'/'others_similarity_value'
        readme_vector: représentation avec le readme du repo pour lequel on cherche les repos les plus similaires
        others_vector: représentation sans le readme du repo pour lequel on cherche les repos les plus similaires
        df_vectors est un dataframe avec trois champs 'id', '{staging}_readme_vector', '{staging}_others_vector'
        '''
        staging = self._staging
        
        #datas['readme_similarity_value'] = datas['readme_vector'].apply(lambda vec: compute_similarity_cosinus(vec, repo_readme_vector) if len(vec) > 0 else 0)
        df_vectors['similarity_score_with_readme'] = df_vectors[f'{staging}_readme_vector'].apply(lambda vector: compute_similarity_cosinus(vector, readme_vector) if len(vector) > 0 else 0)
        df_vectors['similarity_score_with_others'] = df_vectors[f'{staging}_others_vector'].apply(lambda vector: compute_similarity_cosinus(vector, others_vector) if len(vector) > 0 else 0)

        k_repos_most_similar_with_readme = df_vectors.sort_values(by='similarity_score_with_readme', ascending=False).head(k)
        k_repos_most_similar_with_others = df_vectors.sort_values(by='similarity_score_with_others', ascending=False).head(k)

        # On nettoye pour ne garder qu'une colonne vector
        #top_k_readme['vector'] = top_k_readme['readme_vector']
        #top_k_others['vector'] = top_k_others['others_vector']

        # On ne garde que certaines colonnes
        #top_k_readme = top_k_readme[['id', 'vector', 'readme_similarity_value']]
        k_repos_most_similar_with_readme = k_repos_most_similar_with_readme[['id', f'{staging}_readme_vector']]
        #top_k_others = top_k_others[['id', 'vector', 'others_similarity_value']]
        k_repos_most_similar_with_others = k_repos_most_similar_with_others[['id', f'{staging}_others_vector']]

        return k_repos_most_similar_with_readme, k_repos_most_similar_with_others

    def get_top_k_for_evaluation(self, k, repo, df_vectors):
        '''
        k          : nombre de repos à retourner
        repo       : répertoire dont on passe les représentations en entrée '{staging}_readme_vector' et '{staging}_others_vector'
        df_vectors : df contenant les répertoires vectorisés avec trois champs 'id', '{staging}_readme_vector', '{staging}_others_vector' --> répertoires de train

        retourne ???
        '''
        readme_vector = repo[f'{self._staging}_readme_vector']
        others_vector = repo[f'{self._staging}_others_vector']

        res = self._get_top_k(k, readme_vector, others_vector, df_vectors)

        k_repos_most_similar_with_readme = res[0][f'{self._staging}_readme_vector'].to_list()
        k_repos_most_similar_with_others = res[1][f'{self._staging}_others_vector'].to_list()
        
        return k_repos_most_similar_with_readme, k_repos_most_similar_with_others

    def get_top_k_for_prediction(self, k, repo, df_vectors):
        '''
        k          : nombre de repos à retourner
        repo       : répertoire dont on passe les features en entrée 'readme_preproc' et 'others_preproc'
        df_vectors : df contenant les répertoires vectorisés avec trois champs 'id', '{staging}_readme_vector', '{staging}_others_vector' --> tous les répertoires (dans le cadre de l'inférence on utilise toute la base)

        retourne ???
        '''
        readme_vector = self._doc2vec_readme.predict(repo['readme_preproc'])
        others_vector = self._doc2vec_others.predict(repo['others_preproc'])

        return self._get_top_k(k, readme_vector, others_vector, df_vectors)