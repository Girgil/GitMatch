// Connexion à la base MongoDB
db = db.getSiblingDB("github");

// Création de la collection des dépôts GitHub
db.createCollection("repertoires_vecs");
db.createCollection("repertoires_features");
db.createCollection("users");

// Création d’un index sur le champ "name" pour accélérer les recherches
// db.repos.createIndex({ name: 1 }, { unique: true });
db.users.createIndex({ id : 1 }, { unique: true });
db.users.insert({ id: -1 }) // exemple user

db.repertoires_vecs.createIndex({ id: 1 }, { unique: true });
db.repertoires_vecs.insert({ id: -1 }); // exemple repertoire vec

db.repertoires_features.createIndex({ id: 1 }, { unique: true });
db.repertoires_features.insert({ id: -1, readme_preproc: "['test', 'mot']", others_preproc: "['design', 'python']", url: "https://api.github.com/repos/seek-oss/playroom" })


print("✅ Bases de données et index MongoDB initialisés !");
