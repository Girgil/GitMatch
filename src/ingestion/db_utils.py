from pymongo.collection import Collection

def update_users(owners_to_repo, collection: Collection, partition):
    for owner_id, repo_ids in owners_to_repo.items():
        criteria = {'user_id': owner_id}

        # Utiliser update_one avec upsert pour insérer ou mettre à jour
        collection.update_one(
            criteria,
            {
                '$setOnInsert': {'partition': partition},
                '$addToSet': {'rep_id': {'$each': repo_ids}}
            },
            upsert=True
        )