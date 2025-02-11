import os
import openai
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Optional, Tuple, Dict, List, NamedTuple, Set
import scipy
import time
from pprint import pprint as pprint


import tkinter as tk
from tkinter import ttk

'''
tkinter 基本控件

tkinter 提供了许多基本的 GUI 控件（widgets），例如：
Tk：创建主窗口（主应用程序）。
Label：显示文本或图像。
Button：创建按钮。
Entry：单行文本输入框。
Text：多行文本输入框。
Canvas：绘图区域。
Frame：容器，用于组织其他控件。
这些控件可以用来构建各种 GUI 应用程序。
ttk 的增强控件

ttk 提供了一些增强控件，例如：
ttk.Button：主题化按钮。
ttk.Label：主题化标签。
ttk.Entry：主题化单行文本框。
ttk.Progressbar：进度条。
ttk.Treeview：树形视图控件。
ttk.Combobox：下拉列表框。
这些控件的外观更符合现代系统的设计风格
'''





from basic_utils import *
from knowledge_graph import *
from knowledge_graph_querying import *
from initial_card_processing import *

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


model_chat_engine = "gpt-4o" 

SYSTEM_MESSAGE = ("You are a helpful professor and polymath scientist. You want to help a fellow researcher learn more about the world. "
                  + "You are clear, concise, and precise in your answers, and you follow instructions carefully.")



def _gen_chat_response(prompt='hi'):
    response = client.chat.completions.create(
        model=model_chat_engine,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            ],
        temperature=0.7,
        
        )
    message = response.choices[0].message

    return message['content']

def gen_chat_response(prompt='hi'):
    prompt_succeeded = False
    wait_time = 0.1
    while not prompt_succeeded:
        try:
            response = _gen_chat_response(prompt)
            prompt_succeeded = True
        except:
            print('  LM response failed. Server probably overloaded. Retrying after ', wait_time, ' seconds...')
            time.sleep(wait_time)
            wait_time += wait_time*2  # exponential backoff 
    return response





def convert_abstraction_group_to_concept_list(abs_grp):
    '''
    将抽象组转为概念列表
    '''
    concept_list = set()
    
    [concept_list.update(concepts) for concepts in abs_grp.values()]
    
    return list(concept_list)