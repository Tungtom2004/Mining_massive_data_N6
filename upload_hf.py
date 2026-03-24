import os 
from dotenv import load_dotenv
from huggingface_hub import login,HfApi

load_dotenv()
login(os.getenv("HF_TOKEN"))

api = HfApi()

api.upload_large_folder(
    folder_path = '/Volumes/workspace/default/telegram_project/processed/messages_clean_parts_from_db_channels11/',
    repo_id = 'Tungtom2004/mining_massive_data_N6_channel11',
    repo_type = 'dataset'
)

print('Finish')
