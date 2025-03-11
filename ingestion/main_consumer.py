from ConsumerCustom import ConsumerCustom
from dotenv import load_dotenv
from utils import read_config
from pymongo import MongoClient

if __name__ == "__main__":
    client = MongoClient('localhost', 27017)
    features_collection = client['github']['repertoire_features']
    users_collection = client['github']['users']
    
    config = read_config()
    
    env_path = '../.env'
    load_dotenv(env_path)
    token = os.getenv("TOKEN_GITHUB")

    consumer = ConsumerCustom()
    
    consumer.consume(
        token=token,
        feature_collection=features_collection,
        user_collection=users_collection,
        config=config,
        topic='gitmatch',
    )