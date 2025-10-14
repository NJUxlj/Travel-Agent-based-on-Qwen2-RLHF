# 导入一些必要的包
import sys
import os
import json

from typing import List, Any, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


class CPTTrainer:
    '''
    用来在大量的文本上做继续预训练
    '''
    def __init__(self):
        pass