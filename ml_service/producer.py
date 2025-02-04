from confluent_kafka import Producer
import requests
import json

producer = Producer({"bootstrap.servers": "kafka:9092"})
KAFKA_TOPIC = "github_repos"

def fetch_github_repos(query="machine learning"):
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars"
    response = requests.get(url)
    return response.json()["items"]

repos = fetch_github_repos()
for repo in repos:
    producer.produce(KAFKA_TOPIC, json.dumps(repo))
    producer.flush()

print("📤 Dépôts envoyés à Kafka")
