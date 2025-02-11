import os
import openai

from openai.types.chat import ChatCompletion
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Optional, Tuple, Dict, List, NamedTuple, Set
import scipy
import time
from pprint import pprint as pprint


# Load your API key from an environment variable or secret management service
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") )

model_to_use =  "gpt-4o"




# Setup interface with language model

def gen_response(prompt='hi', max_tokens=10, temperature=0)->ChatCompletion:
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model_to_use, 
        # prompt=prompt, 
        messages = messages,
        temperature=temperature, 
        max_tokens=max_tokens
    )
    return response

def gen_response_text_with_backoff(prompt='hi', max_tokens=10, temperature=0):
    prompt_succeeded = False
    wait_time = 0.1
    while not prompt_succeeded:
        try:
            response = gen_response(prompt, max_tokens, temperature=temperature)
            prompt_succeeded = True
        except:
            print('  LM response failed. Server probably overloaded. Retrying after ', wait_time, ' seconds...')
            time.sleep(wait_time)
            wait_time += wait_time*2  # exponential backoff 
    response_text = str(response.choices[0].message['content'])
    used_tokens = response.usage["completion_tokens"]  # ["total_tokens"], i think completion is what matters
    return response_text, used_tokens

def get_dict_items_sorted_by_decreasing_value(_dict):
    # 获取字典值的列表，并对其进行排序，得到排序后元素在原列表中的索引
    # 然后反转索引，使得索引对应的值按降序排列
    sort_inds = np.flip(np.argsort(list(_dict.values())))
    # 将字典的键转换为 numpy 数组，并使用降序索引进行索引，得到按值降序排列的键数组
    sorted_keys = np.array(list(_dict.keys()))[sort_inds]
    # 将字典的值转换为 numpy 数组，并使用降序索引进行索引，得到按值降序排列的值数组
    sorted_values = np.array(list(_dict.values()))[sort_inds]
    return sorted_keys, sorted_values
    
def display_dict_sorted_by_decreasing_value(_dict, print_num=10):
    sorted_keys, sorted_values = get_dict_items_sorted_by_decreasing_value(_dict)
    pprint(list(zip(sorted_keys, sorted_values))[0:print_num])