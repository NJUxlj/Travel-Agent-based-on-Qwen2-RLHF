import os
import openai
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Optional, Tuple, Dict, List, Set
import scipy
import time


from basic_utils import *
from knowledge_graph import *


QUESTION_PROMPT = "Question: "
ANSWER_PROMPT = "Answer: "
CONCEPT_LIST_PROMPT = "Extracted key words and concepts: "
RELATED_QUESTION_PROMPT = "Related question: "







def extract_concepts_in_knowledgeGraph_from_subject_list(subject_list, knowledgeGraph):
    pass