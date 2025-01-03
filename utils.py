import os
import torch
import logging
import evaluate
import random
import argparse
import numpy as np
from itertools import product


import datetime


from typing import List, Union, Optional  
import torch.nn as nn
import torch.distributed as dist  
import torch.multiprocessing as mp  
from torch.nn.parallel import DistributedDataParallel as DDP 

from torch.utils.data import DataLoader, DistributedSampler

from datasets import (
    Dataset,
    load_dataset
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaTokenizerFast,
    GPT2TokenizerFast,
    BertTokenizerFast,
    T5TokenizerFast,
    Qwen2TokenizerFast,
    AutoConfig,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    
    BertForSequenceClassification,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    RobertaForSequenceClassification,
    GPT2ForSequenceClassification,
)


from peft import (
    PeftModel,
    
)


from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import os  
import sys




def parse_args():
    parser = argparse.ArgumentParser(description="SFT Trainer Arguments")
    parser.add_argument("--model_name", type=str, required=True, help="基础模型名称")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--device_map", type=str, default="auto", help="设备映射策略")
    
    # 添加 DeepSpeed 所需的参数  
    parser.add_argument("--local_rank", type=int, default=-1)  
    parser.add_argument("--deepspeed", type=str, default=None) 
    
    return parser.parse_args()




def check_deepspeed_env():  
    """检查DeepSpeed环境"""  
    import pkg_resources  
    import torch  
    
    print("\n=== Environment Check ===")  
    print(f"PyTorch version: {torch.__version__}")  
    print(f"CUDA available: {torch.cuda.is_available()}")  
    if torch.cuda.is_available():  
        print(f"CUDA version: {torch.version.cuda}")  
        print(f"GPU count: {torch.cuda.device_count()}")  
    
    try:  
        ds_version = pkg_resources.get_distribution('deepspeed').version  
        print(f"DeepSpeed version: {ds_version}")  
    except pkg_resources.DistributionNotFound:  
        print("DeepSpeed not found!")  
        
    return True








def setup_cuda_debug_environment():  
    """设置调试环境"""  
    import torch  
    
    torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32以获得更精确的错误信息  
    torch.backends.cudnn.deterministic = True      # 使用确定性算法  
    torch.backends.cudnn.benchmark = False         # 禁用基准测试优化  
    
    import os  
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  
    os.environ['TORCH_USE_CUDA_DSA'] = '1' 
    
    print("=== Debug Environment Setup ===")  
    print(f"CUDA available: {torch.cuda.is_available()}")  
    print(f"CUDA version: {torch.version.cuda}")  
    print(f"PyTorch version: {torch.__version__}")  
    print(f"TORCH_USE_CUDA_DSA: {os.getenv('TORCH_USE_CUDA_DSA')}")  
    print(f"Current device: {torch.cuda.current_device()}")  
    print(f"Device name: {torch.cuda.get_device_name()}")  
    print("===========================")  
    
    
    
    
    
    
    
    
    
    
    
    
def get_model_name_using_model(model):
    '''
    
    use the model object's config file to retrieve the model name, e.g. bert-base-uncased
    '''
    
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
        
    config = model.config  
    # 尝试直接获取模型的名称  
    if hasattr(config, 'name_or_path') and config.name_or_path is not None:  
        # 使用 os.path.basename 提取路径中的模型名称  
        model_name = os.path.basename(config.name_or_path)  
        return model_name  
    # 根据模型类型和隐藏层大小推断模型名称  
    if config.model_type == "bert":  
        if config.hidden_size == 768:  
            return "bert-base-uncased"  
        elif config.hidden_size == 1024:  
            return "bert-large-uncased"  
    elif config.model_type == "roberta":  
        if config.hidden_size == 768:  
            return "roberta-base"  
        elif config.hidden_size == 1024:  
            return "roberta-large"  
    elif config.model_type == "llama":  
        if config.hidden_size == 4096:  
            return "meta-llama/Llama-2-13b-hf"  
        elif config.hidden_size == 5120:  
            return "meta-llama/Llama-2-70b-hf"  
    elif config.model_type == "qwen2":  
        if config.hidden_size == 896:  
            return "Qwen2.5-0.5B"  
        elif config.hidden_size == 1536:  
            return "Qwen2.5-1.5B"  
        elif config.hidden_size == 2048:
            return "Qwen2.5-3B"
        elif config.hidden_size == 3584:
            return "Qwen2.5-7B"
    elif config.model_type == "gpt2":
        if config.n_embd == 768:
            return "gpt2"
        elif config.n_embd == 1024:
            return "gpt2-medium"
        elif config.n_embd == 1280:
            return "gpt2-large"
        elif config.n_embd== 1600:
            return "gpt2-xl"
    else:  
        # 无法匹配已知模型，返回未知模型提示  
        raise ValueError("unknown model, please check your config, it should be [bert | llama | qwen2]") 

def get_base_model_using_model(model):
    """
    获取模型包装器的底层的基座模型对象

    """
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
    
        # 获取模型类型  
    model_type = type(model)

    if hasattr(model, "config"):
        config = model.config
    else:
        raise RuntimeError("This model object does not have a config file, check again~~~")

    try:
        if isinstance(model, AutoModel):
            model = model
        elif isinstance(model, PeftModel):  
            print("Info: Model is a PeftModel, getting the base model")  
            model = model.get_base_model() 
        elif isinstance(model, AutoModelForSequenceClassification):
            model = model.base_model
        elif isinstance(model, BertForSequenceClassification):
            model = model.bert
        elif isinstance(model, RobertaForSequenceClassification):
            model = model.roberta
        elif isinstance(model, Qwen2ForSequenceClassification):
            model = model.model
        elif isinstance(model, GPT2ForSequenceClassification):
            model = model.transformer
         
        else:
            raise ValueError(f"the passed model object is not either SequenceClassification model or AutoModel \
                The current model type = {model_type}")

    except:
        raise ValueError(f"Extracting base model failed, your current model type is {model_type}")

    return model

def get_hidden_size_using_config():
    pass

def get_hidden_size_by_model_name(model_name:str):
    pass

def get_hidden_size_using_model(model):
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
    
        # 获取模型类型  
    model_type = type(model)
    
    model_name = get_model_name_using_model(model)

    if hasattr(model, "config"):
        config = model.config
    else:
        raise RuntimeError("This model object does not have a config file, check again~~~")
    
    if hasattr(config,'hidden_size'):
        hidden_size = config.hidden_size
    elif hasattr(config, 'd_model'): # t5
        hidden_size = config.d_model
    elif hasattr(config, 'n_embd'): # gpt2
        hidden_size = config.n_embd
    else:
        raise ValueError(f"the passed model object does not have the attribute `hidden_size` \
            The current model type = {model_type}")
    print(f"model:{model_name}'s hidden_size = {hidden_size}")
    return hidden_size

def get_classifier_from_model(model)-> nn.Module:  
    """  
    获取预训练模型的分类器  
    
    Args:  
        model : AutoModelForSequenceClassification or BertForSequenceClassification
        num_labels (int): 分类标签数量  
    
    Returns:  
        nn.Module: 分类器模块  
    """  
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module

    # 获取分类器  
    if hasattr(model, 'classifier'):  
        # BERT、RoBERTa 等模型的分类器  
        classifier = model.classifier  
        print(f"分类器类型: {type(classifier).__name__}")
        
    elif hasattr(model, 'score'):   # qwen2, gpt2
        # 某些模型可能使用 score 作为分类器名称  
        classifier = model.score  
    else:  
        raise AttributeError("无法找到模型的分类器层")  
    
    # 打印分类器信息  
    print("分类器结构：")  
    print(classifier)  
    
    in_features=None
    out_features=None
    if hasattr(classifier, 'dense'):
        in_features = classifier.dense.in_features
        print("这是一个RobertaClassificationHead，需要通过dense层获取输入维度")
    else:
        in_features = classifier.in_features
        
    if hasattr(classifier, 'out_proj'):
        out_features = classifier.out_proj.out_features
        print("这是一个RobertaClassificationHead，需要通过out_proj层获取输出维度")
    else:
        out_features = classifier.out_features
        
        
    print(f"\n分类器输入维度: {in_features}")  
    print(f"分类器输出维度: {out_features}") 
    
    # 示例：直接使用分类器进行前向传播  
    # batch_size = 4  
    # hidden_size = classifier.in_features  
    
    # 模拟来自BERT的特征输出  
    # dummy_features = torch.randn(batch_size, hidden_size)  
    
    # # 直接使用分类器进行预测  
    # with torch.no_grad():  
    #     outputs = classifier(dummy_features)  
        
    # print(f"\n分类器输出形状: {outputs.shape}")  
    # print("分类器输出示例：")  
    # print(outputs)   
    
    
    print("\n分类器的可训练参数：")  
    for name, param in classifier.named_parameters():  
        print(f"{name}: {param.shape}")  
        
    return classifier 

def get_max_length_from_model(model):  
    """  
    获取模型的最大序列长度  
    model: 既可以base model， 也可以是特定任务model
    
    """  
    if isinstance(model,str):
        model = AutoModel.from_pretrained(model)
    
    # 处理被Accelerator(DDP)包装的模型  
    if hasattr(model, "module"):  
        print("This model is wrapped by Accelerator(DDP), we use model.module")  
        model = model.module  
        
    if hasattr(model, "config"):
        config = model.config  
    else:
        raise ValueError('your model object is not properly defined ... since we can not find a `config` attribute')
    
    # 首先尝试从config中直接获取max_position_embeddings  
    if hasattr(config, 'max_position_embeddings'):  
        return config.max_position_embeddings  
    
    # 如果没有max_position_embeddings，尝试获取max_sequence_length  
    elif hasattr(config, 'max_sequence_length'):  
        return config.max_sequence_length  
    
    elif hasattr(config, 'n_positions'):  
        return config.n_positions
    
    elif hasattr(config, 'n_ctx'):  
        return config.n_ctx
    
    else:
        raise ValueError("Error model object, please check your config, it should have either [max_position_embeddings | max_sequence_length]") 

def get_classifier(model:AutoModelForSequenceClassification):
    """
    获取预训练模型的分类器
    """
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module

    classifier = None
    # 获取分类器
    if hasattr(model, 'classifier'):
        # BERT、RoBERTa 等模型的分类器
        classifier = model.classifier
        print(f"分类器类型: {type(classifier).__name__}")
    elif hasattr(model, 'score'):
        # 某些模型可能使用 score 作为分类器名称
        classifier = model.score
        
    else:
        raise AttributeError("无法找到模型的分类器层")
    
    return classifier

def print_model_info(model:AutoModelForSequenceClassification):  
    """打印模型的详细信息"""  
    
    
    print("\n=== Model Classification Head Information ===")  
    
    # 1. 打印分类器的结构  
    print("\nClassifier Architecture:")  
    if hasattr(model,'classifier'):
        print(model.classifier)  
    elif hasattr(model,'score'):
        print(model.score)
    
    # 2. 打印分类器中dense层的权重形状 
    if hasattr(model,'classifier') and hasattr(model.classifier, 'dense'): 
        dense_weight = model.classifier.dense.weight  
        print("\nDense Layer Weight Shape:", dense_weight.shape)  
    
    # 3. 打印分类器中out_proj层的权重形状  
    if hasattr(model,'classifier') and hasattr(model.classifier, 'out_proj'):
        out_proj_weight = model.classifier.out_proj.weight  
        print("Output Projection Weight Shape:", out_proj_weight.shape)  
    
    # 4. 打印整个模型的参数数量  
    total_params = sum(p.numel() for p in model.parameters())  
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"\nTotal Parameters: {total_params:,}")  
    print(f"Trainable Parameters: {trainable_params:,}")  
    print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%") 

