from confluent_kafka import Consumer
from pymongo import MongoClient
import json

consumer = Consumer({"bootstrap.servers": "kafka:9092", "group.id": "ml-group", "auto.offset.reset": "earliest"})
consumer.subscribe(["github_repos"])

mongo_client = MongoClient("mongodb://mongo:27017/")
db = mongo_client["github"]
collection = db["repos"]

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"❌ Erreur Kafka : {msg.error()}")
        continue

    repo = json.loads(msg.value().decode())
    collection.insert_one(repo)
    print(f"✅ Repo stocké : {repo['name']}")
