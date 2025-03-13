# GitMatch

## Contexte

Projet mené dans le cadre du module Big Data / MLOps du master 2 informatique de l'université du mans par:
- Alex Choux
- Nicolas Giraud
- Anna Béranger

L'objectif de ce projet est de développer une pipeline permettant la <b>recommandation de répertoires github</b>.

## Architecture

L'architecture de notre projet est divisée en 6 répertoires distincts:
- <b>Ingestion</b>: contient les paires producer-consumer pour l'ingestion des données via un cluster kafka externalisé en utilisant confluent kafka;
- <b>Database</b>: contient une surcouche à la base mongodb qui permet de définir un objet qui uniformise les requêtes;
- <b>Mongo</b>: contient la base mongo et tout ce qui est relatif à la sauvegarde des données de manière persistante;
- <b>Model</b>: contient les classes python réalisant l'apprentissage, l'inférence, l'évaluation et le monitoring des modèles;
- <b>Artifacts</b>: contient les traces du versioning des modèles avec mlflow;
- <b>Api</b>: contient tout ce qui est relatif à l'api.

### Ingestion

#### Features
Nous Nous intéressons à des <b>répertoires github pour faire de la recommandation à partir d'un filtrage par contenu</b>. Nous devons donc représenter nos répertoires dans un format permettant un calcul de similarité.  
Nous considérons la représentation sous forme de vecteur en utilisant Doc2Vec.  
Pour apprendre un modèle Doc2Vec il est nécessaire d'avoir une liste de token. Nous devons donc retenir certaines informations des répertoires pour en faire une liste de token à vectoriser.  
Nous considérons les features suivantes: [readme, description, topics et langages]. Nous avons besoin de `contents_url` et de `default_branch` pour pouvoir à terme obtenir le readme. Nous gardons `html_url` pour pouvoir retourner en sortie de l'api des listes d'url.  
#### Structure
Pour réaliser notre ingestion via un topic kafka, nous définissons deux paires découplées de producer - consumer: une paire pour obtenir les données de test, une paire pour obtenir les données d'entraînement.  
La paire destinée à la récupération des données de test recherche des utilisateurs et associe leur identifiant à la liste des répertoires auxquels ils ont contribué. Nous définissons une contribution comme étant un commit, une issue ou une pull-request.   
La paire destinée aux données d'entraînement a deux modes 'created' et 'pushed'.  
Le mode 'created' est à utiliser pour initialiser la base. Il faut définir une date de début et une date de fin. Le producer parcours l'ensemble des jours de l'interval et récupère 100 répertoires pour chaque journée.  
Le mode 'pushed' est à utiliser pour une alimentation continue de la base. Le producer récupère alors 100 répertoires mis à jour une date précise puis se met en attente passive jusqu'au réveil le lendemain.  
Dans les deux paires, le consumer prétraite les répertoires pour obtenir deux listes de token: `readme_preproc` et `others_preproc`. `Readme_preproc` correspond uniquement au prétraitement appliqué sur le readme. `Others_preproc` correspond à l'accumulation de la description pré-traitée, du nom, des topics et des langages.  
Nous enregistrons ces features dans la collection `repertoire_features` avec `html_url`.

### Database
Nous utilisons une base MongoDB à 3 collections: `user`, `repertoire_features` et `repertoire_vectors`.  
La collection `users` fait le lien entre un utilisateur (identifié par son id), une liste d'identifiant des répertoires contribués (commits, issues et pull-requests pour le test; répertoires possédés pour le train).  
La collection `repertoire_features` associe l'identifiant d'un répertoire à ses features prétraitées [readme_preproc, others_preproc] ainsi qu'à son url.  
La collection `repertoire_vectors` associe l'identifiant d'un répertoire à ses représentations vectorielles par les quatre modèles [staging ou production; avec readme ou sans readme]. Le stockage des représentations vectorielles permet de na pas inférer à chaque fois que le système aura besoin d'utiliser les représentations vectorielles des features.

### Model
Nous utilisons Doc2Vec pour modéliser nos vecteurs.  
Nous apprenons deux représentations [avec ou sans le readme] pour deux étapes de vie d'un modèle [staging et production].  
Nous avons en tout quatre modèles disponibles.  
Pour gérer le versioning de ces modèles nous utilisons <b>MLFlow</b> avec les <b>alias</b> et les <b>tags</b>.

### Artifacts
Nous stockons dans artifacts tous les modèles que nous apprenons. L'arborescence permet de retrouver efficacement la version du modèle en production ou en staging.

### API
L'api prend en entrée le nom complet d'un répertoire séparé en deux champs `owner_name` et `repo_name` ainsi qu'un nombre k de repos à recommander.  
Elle retourne deux liste de k urls basée sur les calculs de similarité entre la vectorisation de ce répertoire les vecteurs des répertoires de la base.

## Prérequis
Créer un environnement dans lequel il est nécessaire d'avoir une version de pip ainsi que la version 3.9 de python. Cette version est nécessaire pour assurer la compatibilité avec nos requirements.   
Installer l'environnement requis avec <b>pip install -r requirements</b> depuis un environnement virtuel venv ou conda.

Puis récupérer et installer en-core-web-lg avec:
```bash
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0.tar.gz
pip install en_core_web_lg-3.7.0.tar.gz
```
## Pour dockeriser

La dockerisation complète de la pipeline a été bloqué par la dockerisation du serveur mlflow. Il était impossible d'enregistrer des modèles en se connectant à l'url du serveur mlflow. Par conséquent, seule la bd a été dockerisé. Pour l'activer:
```bash
docker-compose build
docker compose up
```

Cependant, tous les dockerfile ainsi que le docker-compose complet non fonctionnel ont été écrit. Il est possible d'exécuter l'api en se mettant à la racine et en exécutant `python api/app.py`. Elle s'ouve sur l'url suivant `http://localhost:8000/`.

Pour que le projet soit fonctionnel, il faut créer un fichier d'environnement .env à la racine en ajoutant le token d'accès github, ci-joint un [tutoriel](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

Pour utiliser le cluster kafka il faut avoir ajouté, dans  le répertoire `/ingestion`, un fichier `client.properties` structuré selon la norme attendue par un cluster confluent-kafka.

Pour accéder au terminal mongo dans le conteneur : `docker exec -it mongo mongosh`

## Pour tester

Nous mettons à disposition deux notebooks pour tester respectivement l'<b>api</b> et le <b>module de gestion des modèles</b>. Il est <b>nécessaire</b> de dockeriser la base MongoDB en suivant les instructions de la partie précédente.  
Le premier est disponible dans le répertoire `/api` sous le nom `notebook-test-api.ipynb`. Il est <b>nécessaire</b> de lancer l'api en local avant de tester le notebook.  
Le second est situé dans le répertoire `/model` sous le nom `notebook-test-gestion-modele.ipynb`.
