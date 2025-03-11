// Connexion à la base MongoDB
db = db.getSiblingDB("github");

// Création des collections
db.createCollection("repertoire_vectors");
db.createCollection("repertoire_features");
db.createCollection("users");