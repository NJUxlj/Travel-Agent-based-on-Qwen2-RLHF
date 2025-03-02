from src.models.model import TravelAgent  
# from src.ui.app import launch_ui  

from src.finetune.sft_trainer import SFTTrainer

from src.utils.utils import SFTArguments

from src.configs.config import MODEL_PATH, DATA_PATH

from src.data.data_processor import TravelQAProcessor

# 初始化模型  
agent = TravelAgent()  

# # 启动UI  
# launch_ui(agent)


# agent.chat()



args = SFTArguments()  # 使用parse_args获取参数
trainer = SFTTrainer(args = args)

processor = TravelQAProcessor(trainer.tokenizer)

processor.load_dataset_from_hf(DATA_PATH)



processed_data = processor.prepare_training_features()



keys = list(processed_data.keys())

print("keys = ", keys)

trainer.train(
    train_dataset=processed_data["train"][:250],
    eval_dataset=processed_data["train"][250:]
)