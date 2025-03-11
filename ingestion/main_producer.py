from ProducerCustom import ProducerCustom
from dotenv import load_dotenv
from utils import read_config

if __name__ == "__main__":
    config = read_config()
    
    env_path = '../.env'
    load_dotenv(env_path)
    token = os.getenv("TOKEN_GITHUB")

    producer = ProducerCustom(
        curr_date='2015-06-01',
        name_repos_registered=[],
    )
    
    producer.produce(
        topic='gitmatch',
        config=config,
        token=token,
        max_date='2016-06-01',
        file_to_save_repos_name='files_to_save.npy',
        criteria='created'
    )