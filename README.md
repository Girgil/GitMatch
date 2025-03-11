# GitMatch
Projet mené dans le cadre du module Big Data / MLOps du master 2 informatique de l'université du mans par:
- Alex Choux
- Nicolas Giraud
- Anna Béranger

L'objectif de ce projet est de développer une pipeline permettant la recommandation de répertoires github.

L'architecture de notre projet est divisée en 6 répertoires distincts:
- ingestion: contient les paires producer-consumer pour l'ingestion des données via un cluster kafka externalisé en utilisant confluent kafka
- database: contient une surcouche à la base mongodb qui permet de définir un objet qui uniformise les requêtes
- mongo: contient la base mongo et tout ce qui est relatif à la sauvegarde des données de manière persistante
- model: contient les classes python réalisant l'apprentissage, l'inférence, l'évaluation et le monitoring des modèles
- artifacts: contient les traces du versioning des modèles avec mlflow
- api: contient tout ce qui est relatif à l'api

## Ingestion
On s'intéresse à des répertoires github pour faire de la recommandation à partir d'un filtrage par contenu. On doit donc représenter nos répertoires dans un format permettant un calcul de similarité. On considère la représentation sous forme de vecteur en utilisant Doc2Vec. Pour apprendre un modèle Doc2Vec il est nécessaire d'avoir une liste de token. On doit donc retenir certaines informations des répertoires pour en faire une liste de token à vectoriser. On considère les features suivantes: full_name, readme, description, topics et langages. On a besoin du 'contents_url' et 'default_branch' pour pouvoir à terme obtenir le readme. On garde 'html_url' pour pouvoir retourner en sortie de l'api des listes d'url. 
Pour réaliser notre ingestion via un topic kafka, on définit deux paires découplés de producer-consumer: une paire pour les données de test, une paire pour les données de train.
La paire destinée au test cherche des utilisateurs et associe leur identifiant d'utilisateur à la liste des répertoires auxquels ils ont contribué. On définit une contribution comme étant un commit, une issue ou une pull-request. 
La paire destinée au train a deux modes 'created' et 'pushed'. Le mode 'created' est à utiliser pour initialiser la base, le producer récupérera 100 répertoires créés chaque jour entre les date de début et de fin indiquées en paramètres. Le mode 'pushed' est à utiliser pour une alimentation continue de la base, le producer récupérera 100 repos mis à jour à une date précise puis s'endormira jusqu'au lendemain.
Dans les deux paires, le consumer prétraite les répertoires pour obtenir deux listes de token: 'readme_preproc' et 'others_preproc'. 'readme_preproc' correspond uniquement au prétraitement appliqué sur le readme. 'others_preproc' correspond à l'accumulation de la description pré-traitée, du nom, des topics et des langages.
On enregistre ces features dans la collection 'repertoire_features' avec 'html_url'.

## Database
On utilise une base MongoDB à 3 collections: 'user', 'repertoire_features' et 'repertoire_vectors'.
La collection 'users' fait le lien entre un utilisateur (identifié par son id), une liste d'identifiant des répertoires contribués (commits, issues et pull-requests pour le test; répertoires possédés pour le train).
La collection 'repertoire_features' associe l'identifiant d'un répertoire à ses features prétraitées ('readme_preproc', 'others_preproc') ainsi qu'à son url.
La collection 'repertoire_vectors' associe l'identifiant d'un répertoire à ses représentations vectorielles par les quatre modèles (staging ou production; avec readme ou sans readme). Le stockage des représentations vectorielles permet de na pas inférer à chaque fois que le système aura besoin d'utiliser les représentations vectorielles des features.

## Model
On utilise Doc2Vec pour modéliser nos vecteurs. On apprend deux représentations (avec ou sans le readme) pour deux étapes de vie d'un modèle (staging et production). On a en tout quatre modèles disponibles. Pour gérer le versioning de ces modèles on utilises MlFlow avec les alias et les tags.

## Artifacts
On stocke dans artifacts tous les modèles que l'on apprend. L'arborescence permet de retrouver efficacement la version du modèle en production ou en staging.

## API
L'api prend en entrée le nom complet d'un répertoire séparé en deux champs 'owner_name' et 'repo_name' ainsi qu'un nombre k de repos à recommander.
Elle retourne deux liste de k urls basée sur les calculs de similarité entre la vectorisation de ce répertoire les vecteurs des répertoires de la base.

## Prérequis
Installer l'environnement requis avec pip install -r requirements depuis un environnement virtuel venv ou conda.

Puis récupérer en-core-web-lg avec 'python -m spacy download en-core-web-lg'

## Pour dockeriser

La dockerisation complète de la pipeline a été bloqué par la dockerisation du serveur mlflow. Il était impossible d'enregistrer des modèles en se connectant à l'url du serveur mlflow. Par conséquent, seule la bd a été dockerisé. Pour l'activer:
```bash
docker-compose build
docker compose up
```

Cependant, tous les dockerfile ainsi que le docker-cmpose complet non fonctionnel ont été écrit. Il est possible d'exécuter l'api en se mettant à la racine et en tapant 'python api/app.py'. Elle s'ouve sur l'url suivant `http://localhost:8000/`.

Pour que le projet soit fonctionnel, il faut créer un fichier d'environnement .env à la racine en ajoutant le token d'accès github.

Pour utiliser le cluster kafka il faut avoir ajouter un fichier client.properties dans /ingestion.

Pour accéder au terminal mongo dans le conteneur : `docker exec -it mongo mongosh`
