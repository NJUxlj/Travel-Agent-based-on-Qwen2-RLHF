import os, sys
import numpy as np


class SelfPlay:
    def __init__(
        self, 
        game, 
        neural_net, 
        mcts, 
        temperature=1.0):
        
            self.game = game
            self.neural_net = neural_net
            self.mcts = mcts
            self.temperature = temperature
        
    def play_game(self):
        """进行一局自我对弈"""
        training_examples = []
        board = self.game.getInitBoard()
        current_player = 1
        episode_step = 0
        
        while True:
            episode_step += 1
            canonical_board = self.game.getCanonicalForm(board, current_player)
            temp = self.temperature if episode_step < 30 else 0.001
            
            # MCTS搜索
            pi = self.mcts.search(canonical_board)
            pi = self._apply_temperature(pi, temp)
            
            # 记录训练样本
            training_examples.append([
                canonical_board, current_player, pi, None
            ])
            
            # 选择动作
            action = np.random.choice(len(pi), p=pi)
            board, current_player = self.game.getNextState(board, current_player, action)
            
            # 检查游戏是否结束
            game_ended = self.game.getGameEnded(board, current_player)
            if game_ended != 0:
                # 游戏结束，回传最终奖励
                return [(x[0], x[2], game_ended * ((-1) ** (x[1] != current_player))) 
                       for x in training_examples]
    
    def _apply_temperature(self, pi, temperature):
        """应用温度参数"""
        if temperature == 0:
            best_actions = np.where(pi == np.max(pi))[0]
            pi = np.zeros_like(pi)
            pi[np.random.choice(best_actions)] = 1
            return pi
        else:
            pi = np.power(pi, 1.0 / temperature)
            return pi / np.sum(pi)