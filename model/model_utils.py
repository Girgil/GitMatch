import numpy as np

def compute_similarity_cosinus(vec_1, vec_2):
    """
    vec_1: vecteur
    vec_2: vecteur
    retourne le calcul de la similarité cosinus entre les 2 vecteurs
    """
    return np.dot(vec_1, vec_2)/ (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))