import yaml  
from pathlib import Path  
import os

def load_config(config_path: str) -> dict:  
    with open(config_path, 'r', encoding='utf-8') as f:  
        return yaml.safe_load(f)  

# 加载配置  
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_CONFIG = load_config(BASE_DIR / 'configs' / 'model_config.yaml')  
TRAIN_CONFIG = load_config(BASE_DIR / 'configs' / 'train_config.yaml')  
DPO_CONFIG = load_config(BASE_DIR / 'configs' / 'dpo_config.yaml')





BATCH_SIZE = 8
NUM_PROCESSES = 2


OUTPUT_DIR = "output/"
MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-0.5B"

EMBEDDING_MODEL_PATH = "/root/autodl-tmp/models/all-MiniLM-L6-v2"

EMBEDDING_MODEL_PATH_BPE = "D:\\models\\bge-small-en-v1.5"

SFT_MODEL_NAME = "qwen2_sft"
SFT_MODEL_PATH = os.path.join(OUTPUT_DIR, SFT_MODEL_NAME)


DPO_MODEL_NAME = "qwen2_dpo"
DPO_MODEL_PATH = os.path.join(OUTPUT_DIR, DPO_MODEL_NAME)

SFT_DPO_MODEL_NAME = "qwen2_sft_dpo"
SFT_DPO_MODEL_PATH = os.path.join(OUTPUT_DIR, SFT_DPO_MODEL_NAME)



DATA_PATH = "src/data/travel_qa"
RAG_DATA_PATH = "src/data/crosswoz-sft"

DEEPSPEED_CONFIG_PATH = "src/configs/ds_config.json"

PDF_FOLDER_PATH = "src/agents/travel_knowledge/tour_pdfs"
PAGE_FOLDER_PATH = "src/agents/travel_knowledge/tour_pages"

# # 在代码中使用配置  
# model = TravelAgent(  
#     model_name=model_config['model']['name'],  
#     lora_config=model_config['lora']  
# )  

# trainer = Trainer(  
#     model=model,  
#     args=TrainingArguments(**train_config['training'])  
# )  

# dpo_trainer = TravelAgentDPOTrainer(  
#     model=model,  
#     config=DPOConfig(**dpo_config['dpo']['training'])  
# )  