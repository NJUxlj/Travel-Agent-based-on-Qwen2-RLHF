import torch  
import os  
from torch.utils.data import Dataset  
from transformers import (  
    AutoTokenizer,  
    Qwen2ForCausalLM,  
    TrainingArguments,  
    Trainer,  
    TrainerCallback,  
    BitsAndBytesConfig  
)  
from peft import LoraConfig, get_peft_model  
from datasets import load_dataset  
from typing import Optional  
import torch.nn.functional as F  

class CustomGRPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "group_labels": self.data["group_labels"][idx]  
        }  

class GRPOTrainer:  
    def __init__(  
        self,  
        output_dir: str,  
        dataset_name_or_path: str,  
        model_name: str = MODEL_PATH,  
        is_ds: bool = True,  
        ds_config_path: Optional[str] = None,  
        is_peft: bool = True,  
        peft_config: Optional[LoraConfig] = None,  
        is_quantized: bool = False,  
        bnb_config: Optional[BitsAndBytesConfig] = None,  
        max_seq_length: int = 1024,  
        beta: float = 0.1,  
        group_size: int = 4  
    ):  
        self.output_dir = output_dir  
        self.dataset_name_or_path = dataset_name_or_path  
        self.beta = beta  
        self.max_seq_length = max_seq_length  
        self.group_size = group_size  

        # 初始化模型和tokenizer  
        self.model, self.tokenizer = self._init_model_and_tokenizer(  
            model_name, is_quantized, bnb_config  
        )  
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备数据集  
        self.dataset = self._prepare_dataset()  

        # 配置训练参数  
        self.training_args = TrainingArguments(  
            output_dir=output_dir,  
            deepspeed=ds_config_path if is_ds else None,  
            per_device_train_batch_size=4,  
            gradient_accumulation_steps=2,  
            learning_rate=2e-5,  
            bf16=True,  
            logging_steps=10,  
            save_steps=500,  
            remove_unused_columns=False,  
            optim="adamw_torch",  
            max_grad_norm=0.3,  
            num_train_epochs=3  
        )  

        # 初始化自定义Trainer  
        self.trainer = Trainer(  
            model=self.model,  
            args=self.training_args,  
            train_dataset=self.dataset,  
            data_collator=self.grpo_collator,  
            compute_metrics=self._compute_metrics,  
            callbacks=[GRPOCallback(beta=self.beta)]  
        )  

    def _init_model_and_tokenizer(self, model_name, is_quantized, bnb_config):  
        bnb_config = bnb_config or BitsAndBytesConfig(  
            load_in_4bit=True,  
            bnb_4bit_quant_type="nf4",  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
        ) if is_quantized else None  

        tokenizer = AutoTokenizer.from_pretrained(model_name)  
        tokenizer.pad_token = tokenizer.eos_token  

        model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            quantization_config=bnb_config,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        return model, tokenizer  

    def _default_lora_config(self):  
        return LoraConfig(  
            r=64,  
            lora_alpha=16,  
            lora_dropout=0.05,  
            target_modules=["q_proj", "v_proj"],  
            bias="none",  
            task_type="CAUSAL_LM"  
        )  

    def _prepare_dataset(self):  
        dataset = load_dataset(self.dataset_name_or_path, split="train")  
        dataset = dataset.filter(self._data_filter)  
        
        tokenized_data = dataset.map(  
            self._tokenize_function,  
            batched=True,  
            num_proc=4,  
            remove_columns=dataset.column_names  
        )  
        
        return CustomGRPODataset(tokenized_data)  

    def _data_filter(self, sample):  
        return all([sample["prompt"], sample["answer"]]) and \
               len(sample["prompt"]) <= 512 and \
               len(sample["answer"]) <= 1024  

    def _tokenize_function(self, samples):  
        batch = {"input_ids": [], "attention_mask": [], "group_labels": []}  
        
        for prompt, answer in zip(samples["prompt"], samples["answer"]):  
            full_prompt = f"Question: {prompt}\nAnswer: {answer}"  
            
            tokens = self.tokenizer(  
                full_prompt,  
                max_length=self.max_seq_length,  
                padding="max_length",  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            batch["input_ids"].append(tokens["input_ids"][0])  
            batch["attention_mask"].append(tokens["attention_mask"][0])  
            batch["group_labels"].append(torch.tensor([1 if answer == samples["ground_truth"] else 0]))  
            
        return batch  

    def grpo_collator(self, features):  
        batch = {  
            "input_ids": torch.stack([f["input_ids"] for f in features]),  
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),  
            "group_labels": torch.stack([f["group_labels"] for f in features])  
        }  
        return batch  

    def _compute_metrics(self, eval_pred):  
        logits = eval_pred.predictions  
        accuracy = (logits > 0.5).float().mean()  
        return {"grpo_accuracy": accuracy}  

    def train(self):  
        self.trainer.train()  

    def save_model(self):  
        save_path = os.path.join(self.output_dir, "qwen2_grpo")  
        self.trainer.save_model(save_path)  
        self.tokenizer.save_pretrained(save_path)  

class GRPOCallback(TrainerCallback):  
    def __init__(self, beta=0.1):  
        self.beta = beta  
        
    def on_train_begin(self, args, state, control, **kwargs):  
        self.model = kwargs.pop("model")  
        if isinstance(self.model, DeepSpeedEngine):  
            self.model = self.model.module  

    def on_step_begin(self, args, state, control, **kwargs):  
        if state.global_step == 0:  
            self.ref_model = self._clone_model(self.model)  
            self.ref_model.requires_grad_(False)  

    def _clone_model(self, model):  
        return type(model)(**model.config.to_dict()).load_state_dict(model.state_dict())  

    def compute_loss(self, model, inputs, return_outputs=False):  
        outputs = model(  
            input_ids=inputs["input_ids"],  
            attention_mask=inputs["attention_mask"]  
        )  
        
        # 计算当前策略的log概率  
        log_probs = self._get_log_probs(outputs.logits, inputs["input_ids"])  
        
        # 计算参考策略的log概率  
        with torch.no_grad():  
            ref_outputs = self.ref_model(  
                input_ids=inputs["input_ids"],  
                attention_mask=inputs["attention_mask"]  
            )  
            ref_log_probs = self._get_log_probs(ref_outputs.logits, inputs["input_ids"])  
        
        # 计算相对优势  
        advantages = (log_probs - ref_log_probs).view(-1, self.group_size)  
        group_advantages = advantages.mean(dim=1, keepdim=True)  
        
        # GRPO损失函数  
        losses = -F.logsigmoid(self.beta * (group_advantages - advantages))  
        
        return losses.mean()  

    def _get_log_probs(self, logits, labels):  
        log_probs = F.log_softmax(logits, dim=-1)  
        return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)