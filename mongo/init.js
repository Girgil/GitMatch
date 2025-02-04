// Connexion à la base MongoDB
db = db.getSiblingDB("github");

// Création de la collection des dépôts GitHub
db.createCollection("repos");

// Création d’un index sur le champ "name" pour accélérer les recherches
db.repos.createIndex({ name: 1 }, { unique: true });

// Ajout de quelques dépôts de test (optionnel)
db.repos.insertMany([
    { name: "tensorflow", description: "An open source machine learning framework" },
    { name: "scikit-learn", description: "Machine learning in Python" },
    { name: "pytorch", description: "An open source deep learning platform" }
]);

print("✅ Base de données et index MongoDB initialisés !");
