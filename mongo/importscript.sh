#!/bin/bash
echo "Import des données"

mongoimport --db='github' --collection='repertoire_features' --file='/data/repertoire_features.json'
mongoimport --db='github' --collection='users' --file='/data/users.json'

echo "Fin de l'import des données"