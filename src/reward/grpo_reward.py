# 导入一些必要的包
import sys
import os
import json

from typing import List, Any, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import jieba
from rapidfuzz import fuzz
from editdistance import eval as editdistance
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class GRPOReward:
    def __init__(self):
        pass

    def sentence_similarity_reward(self, completions, ground_true_answers, **kwargs):
        '''
        基于句子相似度的奖励函数

        Args:
            completions: 生成的答案: <think></think><answer></answer>
            ground_true_answers: 真实的答案
        
        Returns:
            rewards: 奖励 1, 0, -1

            1: 相似度>= 0.8
            0: 相似度< 0.8
            -1: completion 不符合推理格式， 或者无法解析出答案

            任意句子长度<20, 编辑距离， 否则就用 jaccard 相似度

        '''
        rewards = []
        for completion, ground_true_answer in zip(completions, ground_true_answers):
            completion_answer = completion.split('<answer>')[1].split('</answer>')[0]
            ground_true_answer = ground_true_answer.split('<answer>')[1].split('</answer>')[0]
            reward = self.sentence_similarity_reward(completion_answer, ground_true_answer)
            rewards.append(reward)
        print(f"The rewards for the sentence similarity reward are: {rewards}")
        return rewards


    

    def gpt_verifier_reward():
        pass


    def bleu_rougel_reward():
        pass


    def mcts_rag_reward():
        '''
        结合蒙特卡洛树搜索和 RAG 的奖励函数

        在 RAG 召回的片段中进行 MCTS， 找到奖励最高的那个片段组合，然后喂给 gpt
        '''