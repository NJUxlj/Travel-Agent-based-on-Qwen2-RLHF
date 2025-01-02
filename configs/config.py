import yaml  
from pathlib import Path  

def load_config(config_path: str) -> dict:  
    with open(config_path, 'r', encoding='utf-8') as f:  
        return yaml.safe_load(f)  

# 加载配置  
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_CONFIG = load_config(BASE_DIR / 'configs' / 'model_config.yaml')  
TRAIN_CONFIG = load_config(BASE_DIR / 'configs' / 'train_config.yaml')  
DPO_CONFIG = load_config(BASE_DIR / 'configs' / 'dpo_config.yaml')




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