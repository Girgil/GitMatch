import numpy as np
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model.model_utils import compute_similarity_cosinus

class Evaluation():

    def __init__(self, staging_model, production_model):
        self._staging_model = staging_model
        self._production_model = production_model

    def _get_k_repos_most_similar(self, test_chosen_vectors, staging_vectors, production_vectors, k):
        '''
        test_chosen_vectors: liste qui contient pour chaque utilisateur de test un dict à 4 champs avec les 4 vecteurs

        staging_vectors: df qui contient pour tous les repos de train leurs représentations vectorielles pour le staging et leur id

        production_vectors: df qui contient pour tous les repos de train leurs représentations vectorielles pour la production et leur id
        
        Méthode qui récupère pour chaque utilisateur de test, pour chaque représentation (readme/others) et pour chaque modèle (production/staging) les représentations vectorielles les plus similaires
        '''
        staging_k_repos_most_similar_with_readme = []
        staging_k_repos_most_similar_with_others = []
        production_k_repos_most_similar_with_readme = []
        production_k_repos_most_similar_with_others = []
        
        for test_vector in test_chosen_vectors:
            # k repos pour staging
            staging_test_vector = {
                'staging_readme_vector': test_vector['staging_readme_vector'],
                'staging_others_vector': test_vector['staging_others_vector'],
            }
            
            res = self._staging_model.get_top_k_for_evaluation(k, staging_test_vector, staging_vectors)
            staging_k_repos_most_similar_with_readme.append(res[0])
            staging_k_repos_most_similar_with_others.append(res[1])
    
            # k repos pour production
            production_test_vector = {
                'production_readme_vector': test_vector['production_readme_vector'],
                'production_others_vector': test_vector['production_others_vector'],
            }
            
            res = self._production_model.get_top_k_for_evaluation(k, production_test_vector, production_vectors)
            production_k_repos_most_similar_with_readme.append(res[0])
            production_k_repos_most_similar_with_others.append(res[1])

        return staging_k_repos_most_similar_with_readme, staging_k_repos_most_similar_with_others, production_k_repos_most_similar_with_readme, production_k_repos_most_similar_with_others
    
    def _evaluate_1(self, test_remaining_vectors, k_vectors_most_similar, random_chosen_train_vector):
        '''
        test_remaining_vectors: représentations vectorielles des répertoires mis de côté pour l'utilisateur de test
        --> liste
        
        k_vectors_most_similar: représentations vectorielles des répertoires les plus similaraires par rapport à un vecteur de l'utilisateur de test choisi aléatoirement
        --> liste
        
        random_chosen_train_vector: représentation vectorielle du répertoire de train choisi aléatoirement
        
        Calcule les scores d'évaluation de la première méthode d'évaluation
        La méthode consiste à récupérer les k repos les plus similaires avec le répertoire de l'utilisateur de test
        Pour chaque repo parmi les k, on calcule sa similarité moyenne avec les repos mis de côté de l'utilisateur de test
        On calcule la moyenne de ces similarités ce qui donne un score s1
        On a tiré aléatoirement un repo d'entraînement
        On calcule sa similarité moyenne avec les repos mis de côté, on obtient un score s2
        On retourne (s1 - s2)
        '''

        # Calcul de la moyenne des similarités moyennes de chacun des k repos avec les repos de test
        mean_similarities = []

        for vector in k_vectors_most_similar:
            vector_similarities = [compute_similarity_cosinus(vector, test_vector) for test_vector in test_remaining_vectors]
            mean_similarities = np.mean(vector_similarities)

        s1 = np.mean(mean_similarities)

        # Calcul la similarité moyenne du répertoire de train tiré aléatoirement
        s2 = np.mean([compute_similarity_cosinus(random_chosen_train_vector, test_vector) for test_vector in test_remaining_vectors])
        
        evaluation_score_1 = s1 - s2
        
        return evaluation_score_1

    def _evaluate_1_for_all_test_users(self, users_test_remaining_vectors, users_k_vectors_most_similar, users_random_chosen_train_vector):
        '''
        users_test_remaining_vectors: liste qui contient pour chaque utilisateur de test les représentations vectorielles des répertoires qui ont été mis de côté
        
        users_k_vectors_most_similar: liste qui contient pour chaque utilisateur de test les représentations vectorielles des répertoires de train les plus similaires au répertoire de test qui a été choisi aléatoirement
        
        users_random_chosen_train_vector: liste qui contient pour chaque utilisateur de test la représentation vectorielle d'un répertoire de train qui va nous permettre de nous comparer à la prédiction aléatoire
        --> liste de dict

        Calcul le score moyen d'un modèle (readme ou others, staging ou prod) pour l'ensemble des utilisateurs de test
        La méthodologie d'évaluation est expliquéé dans _evaluate_1

        On obtient un score moyen mean_similarity qu'on retourne
        '''

        mean_similarity = np.mean([self._evaluate_1(test_remaining_vectors, users_k_vectors_most_similar[ii], users_random_chosen_train_vector[ii]) for ii, test_remaining_vectors in enumerate(users_test_remaining_vectors)])

        return mean_similarity

    def _evaluate_2(self, test_remaining_vectors, three_vectors_among_most_similar):
        '''
        test_remaining_vectors: représentations vectorielles des répertoires d'un utilisateur de test mis de côté
        --> liste

        three_vectors_among_most_similar: représentations vectorielles des trois répertoires parmi les k plus similaires au répertoire choisi aléatoirement pour un utilisateur de test
        --> list
        
        Calcule les scores d'évaluation de la seconde méthode d'évaluation

        Prend en entrée trois représentations vectorielles de répertoires parmi les k plus similaires au répertoire choisi aléatoirement pour un utilisateur de test

        Les 3 vecteurs correspondent au répertoire le plus similaire du classement (idx=0), au répertoire au milieu du classement (idx=k//2) et au répertoire le moins similaire du classement (idx=k-1)

        Pour chacun de ces vecteurs on calcule la similarité moyenne avec les répertoires mis de côté pour les utilsateurs de test on obtient trois scores s1 (idx=0), s2 (idx=k//2), s3 (idx=k-1)

        Le score d'évaluation correspond à la somme des différences entre les scores des répertoires : (s1 - s2) + (s2 - s3)
        L'objectif est d'évaluer la qualité du tri
        '''
        
        mean_similarities = []

        for vector in three_vectors_among_most_similar:
            similarities = [compute_similarity_cosinus(vector, test_vector) for test_vector in test_remaining_vectors]
            mean_similarities.append(np.mean(similarities))
        
        evaluation_score_2 = (mean_similarities[0] - mean_similarities[1]) + (mean_similarities[1] - mean_similarities[2])
        
        return evaluation_score_2

    def _evaluate_2_for_all_test_users(self, users_test_remaining_vectors, users_three_vectors_among_most_similar):
        '''
        users_test_remaining_vectors: liste qui contient pour chaque utilisateur de test les représentations vectorielles des répertoires qui ont été mis de côté
        
        users_three_vectors_among_most_similar: liste qui contient pour chaque utilisateur de test les représentations vectorielles de trois répertoires de train parmi les k plus similaires au répertoire de test qui a été choisi aléatoirement selon l'indice 0, k//2 et k-1

        Calcul le score moyen d'un modèle (readme ou others, staging ou prod) pour l'ensemble des utilisateurs de test
        La méthodologie d'évaluation est expliquéé dans _evaluate_2

        On obtient un score moyen mean_similarity qu'on retourne
        '''

        mean_similarity = np.mean([self._evaluate_2(test_remaining_vectors, users_three_vectors_among_most_similar[ii]) for ii, test_remaining_vectors in enumerate(users_test_remaining_vectors)])

        return mean_similarity

    def evaluate(self, staging_vectors, production_vectors, test_chosen_vectors, test_remaining_vectors, k):
        '''
        staging_vectors: df contenant 3 champs 'id', 'staging_readme_vector', 'staging_others_vector' pour les données de train

        production_vectors: df contenant 3 champs 'id', 'production_readme_vector', 'production_others_vector' pour les données de train

        test_chosen_vectors: représentations vectorielles du répertoire choisi pour chaque utilisateur de test
        --> liste de dict à 4 champs 'st_rd_v', 'st_ot_v', 'pd_rd_v', 'pd_ot_v'

        test_remaining_vectors: représentations vectorielles des répertoires des utilisateurs mis de côté
        --> liste de df avec les 4 champs au dessus

        k: nombre de repos similaires qu'on considère
        
        Méthode qui calcule pour les modèles en staging et en production des scores d'évaluation à partir des 2 méthodes
        Elle retourne 2 listes staging_scores et production_scores
        Ces 2 listes sont formatées comme il suit : [score1 pour readme, score1 pour others, score2 pour readme, score2 pour others]
        '''

        number_of_test_users = len(test_chosen_vectors)
        
        # Inférence pour récupérer les k vecteurs les plus similaires pour chaque utilisateur de test
        res = self._get_k_repos_most_similar(test_chosen_vectors, staging_vectors, production_vectors, k)

        staging_k_repos_most_similar_with_readme = res[0]
        staging_k_repos_most_similar_with_others = res[1]
        
        production_k_repos_most_similar_with_readme = res[2]
        production_k_repos_most_similar_with_others = res[3]
        
        # Choix aléatoire d'un répertoire de train pour chaque utilisateur de test pour l'évaluation de la méthode 1
        train_chosen_repos_id = []

        # On considère seulement les répertoires qui ont une représentation utilisant un readme
        filtered_staging_vectors = staging_vectors[staging_vectors['staging_readme_vector'].apply(lambda x: len(x) > 0)]
        
        for _ in range(number_of_test_users):
            train_chosen_repos_id.append(filtered_staging_vectors.sample(n=1)['id'].values[0])

        # Création des listes pour la première méthode d'évaluation
        staging_readme_chosen_vectors = []
        staging_others_chosen_vectors = []
        production_readme_chosen_vectors = []
        production_others_chosen_vectors = []

        for repo_id in train_chosen_repos_id:
            #staging_df[staging_df['id'] == repo_id]['readme_vector'].values[0]
            staging_readme_chosen_vectors.append(staging_vectors[staging_vectors['id'] == repo_id]['staging_readme_vector'].values[0])
            staging_others_chosen_vectors.append(staging_vectors[staging_vectors['id'] == repo_id]['staging_others_vector'].values[0])
            production_readme_chosen_vectors.append(production_vectors[production_vectors['id'] == repo_id]['production_readme_vector'].values[0])
            production_others_chosen_vectors.append(production_vectors[production_vectors['id'] == repo_id]['production_others_vector'].values[0])

        # Passage pour les utilisateurs de test d'une liste de df à 4 listes de liste pour chacune des 4 représentations vectorielles (rd/ot, stg/prd)
        staging_readme_test_remaining_vectors = []
        staging_others_test_remaining_vectors = []
        production_readme_test_remaining_vectors = []
        production_others_test_remaining_vectors = []

        for df in test_remaining_vectors:
            staging_readme_test_remaining_vectors.append(df['staging_readme_vector'].to_list())
            staging_others_test_remaining_vectors.append(df['staging_others_vector'].to_list())
            production_readme_test_remaining_vectors.append(df['production_readme_vector'].to_list())
            production_others_test_remaining_vectors.append(df['production_others_vector'].to_list())
        
        # Calcul des scores d'évaluation de la méthode 1
        # test_remaining_vectors, k_vectors_most_similar, random_chosen_train_vector
        staging_readme_score_1 = self._evaluate_1_for_all_test_users(staging_readme_test_remaining_vectors, staging_k_repos_most_similar_with_readme, staging_readme_chosen_vectors)
        staging_others_score_1 = self._evaluate_1_for_all_test_users(staging_others_test_remaining_vectors, staging_k_repos_most_similar_with_others, staging_others_chosen_vectors)
        production_readme_score_1 = self._evaluate_1_for_all_test_users(production_readme_test_remaining_vectors, production_k_repos_most_similar_with_readme, production_readme_chosen_vectors)
        production_others_score_1 = self._evaluate_1_for_all_test_users(production_others_test_remaining_vectors, production_k_repos_most_similar_with_others, production_others_chosen_vectors)

        # Filtrage des k repos similaires pour ne garder que le top (idx=0), mid (idx=k//2) et bottom (idx=k-1)
        filtered_staging_k_repos_most_similar_with_readme = [[k_repos[0], k_repos[k//2], k_repos[-1]] for k_repos in staging_k_repos_most_similar_with_readme]
        filtered_staging_k_repos_most_similar_with_others = [[k_repos[0], k_repos[k//2], k_repos[-1]] for k_repos in staging_k_repos_most_similar_with_others]
        filtered_production_k_repos_most_similar_with_readme = [[k_repos[0], k_repos[k//2], k_repos[-1]] for k_repos in production_k_repos_most_similar_with_readme]
        filtered_production_k_repos_most_similar_with_others = [[k_repos[0], k_repos[k//2], k_repos[-1]] for k_repos in production_k_repos_most_similar_with_others]
        
        # Calcule des scores d'évaluation de la méthode 2
        # users_test_remaining_vectors, users_three_vectors_among_most_similar
        staging_readme_score_2 = self._evaluate_2_for_all_test_users(staging_readme_test_remaining_vectors, filtered_staging_k_repos_most_similar_with_readme)
        staging_others_score_2 = self._evaluate_2_for_all_test_users(staging_others_test_remaining_vectors, filtered_staging_k_repos_most_similar_with_others)
        production_readme_score_2 = self._evaluate_2_for_all_test_users(production_readme_test_remaining_vectors, filtered_production_k_repos_most_similar_with_readme)
        production_others_score_2 = self._evaluate_2_for_all_test_users(production_others_test_remaining_vectors, filtered_production_k_repos_most_similar_with_others)

        staging_scores = [staging_readme_score_1, staging_others_score_1, staging_readme_score_2, staging_others_score_2]
        production_scores = [production_readme_score_1, production_others_score_1, production_readme_score_2, production_others_score_2]

        return np.array(staging_scores), np.array(production_scores)