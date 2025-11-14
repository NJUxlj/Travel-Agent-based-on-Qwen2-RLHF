import os, sys
from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    # 训练参数
    num_iterations = 100
    num_self_play_games = 100
    max_training_examples = 200000
    
    # 神经网络参数
    learning_rate = 0.001
    epochs = 10
    batch_size = 64
    
    # MCTS参数
    c_puct = 1.0
    num_simulations = 800
    temperature = 1.0
    
    # 评估参数
    eval_frequency = 5
    eval_games = 20
    eval_simulations = 400
    update_threshold = 0.55