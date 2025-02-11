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

from basic_utils import *



# Convert dataframe to json and save to file in human readable format. 
def save_cards_df_to_json(cards_df:pd.DataFrame, save_file_name):
    cards_df_as_json = cards_df.to_json(orient = 'index') # orient = 'index' 表示以索引为导向进行转换。
    with open(save_file_name+".json", mode='w') as f:
        json.dump(cards_df_as_json, f)
    

def read_cards_df_from_json(save_file_name):
    with open(save_file_name+".json", "r") as f:
        cards_df_reloaded = pd.read_json(json.load(f), orient="index")

    return cards_df_reloaded

# Generate initial text descriptions from front and back of a flashcard 

def get_cards_df_text_descriptions_from_front_and_back(flashcardExamples_front, 
                                                   flashcardExamples_back,
                                                  verbose=False):
    pass




def get_cards_df_abstraction_groups_from_meta_data(cards_df):
    """
    Converts meta data with weird names for abstraction levels into a common format 
    """




def get_cards_df_meta_data_from_text_descriptions(cards_df_text_descriptions,
                                                  verbose=False):
    """
    Returns a pandas dataframe with the text descriptions converted into dictionaries
    which contain the separated out key words at various levels of abstraction.
    """
    
    
    
    
def get_cards_df_abstraction_groups_from_front_and_back_list(flashcardExamples_front, 
                                                   flashcardExamples_back,
                                                  verbose=False):
    pass