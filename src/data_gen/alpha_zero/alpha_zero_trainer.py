import sys, os
from pathlib import Path
import torch
import numpy as np
from alpha_zero.mcts_search import (
    MCTSNode, MCTS
)

from alpha_zero.self_play import (
    SelfPlay
)

class AlphaZeroTrainer:
    def __init__(self, game, neural_net, args):
        self.game = game
        self.neural_net = neural_net
        self.args = args
        self.mcts = MCTS(neural_net, game, args.c_puct, args.num_simulations)
        self.self_play = SelfPlay(game, neural_net, self.mcts, args.temperature)
        self.training_examples = []
        
    def train(self):
        """主训练循环"""
        for iteration in range(self.args.num_iterations):
            print(f"开始第 {iteration + 1} 轮迭代")
            
            # 自我对弈收集数据
            iteration_examples = []
            for game_num in range(self.args.num_self_play_games):
                examples = self.self_play.play_game()
                iteration_examples.extend(examples)
                
            # 添加到训练集
            self.training_examples.extend(iteration_examples)
            
            # 保持训练集大小
            if len(self.training_examples) > self.args.max_training_examples:
                self.training_examples = self.training_examples[-self.args.max_training_examples:]
            
            # 训练神经网络
            old_neural_net = self._copy_neural_net()
            self._train_neural_net()
            
            # 评估新网络
            if iteration % self.args.eval_frequency == 0:
                win_rate = self._evaluate_neural_net(old_neural_net)
                if win_rate < self.args.update_threshold:
                    print("新网络表现不佳，恢复旧网络")
                    self.neural_net = old_neural_net
                else:
                    print(f"新网络获胜率: {win_rate:.2f}")
                    
    def _train_neural_net(self):
        """训练神经网络"""
        optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=self.args.learning_rate)
        
        for epoch in range(self.args.epochs):
            batch_losses = []
            
            # 随机打乱训练数据
            np.random.shuffle(self.training_examples)
            
            for batch_start in range(0, len(self.training_examples), self.args.batch_size):
                batch = self.training_examples[batch_start:batch_start + self.args.batch_size]
                
                # 准备批次数据
                boards = torch.FloatTensor([ex[0] for ex in batch])
                target_pis = torch.FloatTensor([ex[1] for ex in batch])
                target_vs = torch.FloatTensor([ex[2] for ex in batch])
                
                # 前向传播
                out_pi, out_v = self.neural_net(boards)
                
                # 计算损失
                value_loss = F.mse_loss(out_v.view(-1), target_vs)
                policy_loss = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                total_loss = value_loss + policy_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                batch_losses.append(total_loss.item())
            
            print(f"Epoch {epoch + 1}: Loss = {np.mean(batch_losses):.4f}")
    
    def _evaluate_neural_net(self, old_net):
        """评估新网络vs旧网络"""
        new_mcts = MCTS(self.neural_net, self.game, self.args.c_puct, self.args.eval_simulations)
        old_mcts = MCTS(old_net, self.game, self.args.c_puct, self.args.eval_simulations)
        
        wins = 0
        for game_num in range(self.args.eval_games):
            result = self._play_evaluation_game(new_mcts, old_mcts, game_num % 2)
            if result == 1:
                wins += 1
        
        return wins / self.args.eval_games
    
    def _play_evaluation_game(self, mcts1, mcts2, starting_player):
        """进行评估对局"""
        board = self.game.getInitBoard()
        current_player = 1
        mcts_players = [mcts2, mcts1] if starting_player == 0 else [mcts1, mcts2]
        
        while True:
            current_mcts = mcts_players[current_player == starting_player + 1]
            canonical_board = self.game.getCanonicalForm(board, current_player)
            
            pi = current_mcts.search(canonical_board)
            action = np.argmax(pi)
            
            board, current_player = self.game.getNextState(board, current_player, action)
            game_ended = self.game.getGameEnded(board, current_player)
            
            if game_ended != 0:
                return game_ended * (1 if current_player == starting_player + 1 else -1)