import os 
from dotenv import load_dotenv
from huggingface_hub import login,HfApi

load_dotenv()
login(os.getenv("HF_TOKEN"))

api = HfApi()

api.upload_folder(
    folder_path = '/Volumes/workspace/default/telegram_project/processed/messages_clean_parts/',
    repo_id = 'Tungtom2004/mining_massive_data_N6',
    repo_type = 'dataset'
)

print('Finish')
