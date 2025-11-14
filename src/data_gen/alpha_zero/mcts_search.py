import math
import numpy as np

class MCTSNode:
    def __init__(self, prior_prob, parent=None):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    
    def get_value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def select_child(self, c_puct):
        """使用PUCT公式选择最佳子节点"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            score = self._get_puct_score(child, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def _get_puct_score(self, child, c_puct):
        """计算PUCT分数"""
        q_value = child.get_value()
        exploration_term = c_puct * child.prior_prob * math.sqrt(self.visits) / (1 + child.visits)
        return q_value + exploration_term
    
    def expand(self, action_probs):
        """扩展节点"""
        for action, prob in enumerate(action_probs):
            if prob > 0:
                self.children[action] = MCTSNode(prior_prob=prob, parent=self)
    
    def backup(self, value):
        """回传价值"""
        self.visits += 1
        self.value_sum += value
        if not self.is_root():
            self.parent.backup(-value)  # 对手视角，价值取反



class MCTS:
    def __init__(self, neural_net, game, c_puct=1.0, num_simulations=800):
        self.neural_net = neural_net
        self.game = game
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        
    def search(self, root_state):
        """执行MCTS搜索"""
        root = MCTSNode(prior_prob=1.0)
        
        # 初始扩展根节点
        policy, value = self.neural_net.predict(root_state)
        valid_actions = self.game.getValidMoves(root_state, 1)
        policy = policy * valid_actions  # 屏蔽无效动作
        policy = policy / np.sum(policy)  # 重新归一化
        root.expand(policy)
        
        # 执行模拟
        for _ in range(self.num_simulations):
            self._simulate(root, root_state)
            
        # 返回访问次数分布作为策略
        visits = [0] * self.game.getActionSize()
        for action, child in root.children.items():
            visits[action] = child.visits
            
        return visits
    
    def _simulate(self, node, state):
        """单次模拟"""
        current_node = node
        current_state = state
        search_path = [current_node]
        current_player = 1
        
        # 选择阶段：找到叶子节点
        while not current_node.is_leaf():
            action, current_node = current_node.select_child(self.c_puct)
            current_state, current_player = self.game.getNextState(
                current_state, current_player, action
            )
            search_path.append(current_node)
        
        # 评估阶段
        game_ended = self.game.getGameEnded(current_state, current_player)
        if game_ended != 0:
            # 游戏结束
            value = game_ended * current_player
        else:
            # 使用神经网络评估
            policy, value = self.neural_net.predict(current_state)
            value = value * current_player
            
            # 扩展节点
            valid_actions = self.game.getValidMoves(current_state, current_player)
            policy = policy * valid_actions
            if np.sum(policy) > 0:
                policy = policy / np.sum(policy)
                current_node.expand(policy)
        
        # 回传阶段
        for i, node in enumerate(reversed(search_path)):
            node.backup(value if i % 2 == 0 else -value)