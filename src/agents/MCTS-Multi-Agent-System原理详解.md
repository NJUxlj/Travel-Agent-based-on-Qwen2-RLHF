# 技术栈
- 微软 Agent-Framework

# MASTER [MASTER: A Multi-Agent System with LLM Specialized MCTS] 综述

![alt text](./images/MASTER.png)

Figure 1: Reasoning Tree of MASTER. Starting from Agent0, Agent1 and Agent2 are created in the first expansion. Then the system first selects Agent1 for expansion due to its higher UCT. Its child agent Agent3 is a terminal agent that failed evaluation which triggers a backpropagation and lowers the UCT of Agent1. Now Agent2 has the highest UCT and is selected for next expansion. Its child agent, Agent6 is a terminal agent and passes evaluation. The answer in it is the final answer.





# MASTER 原理


## What is MCTS?
MCTS (Coulom, 2006) is a widely used planning algorithm and famously employed in AlphaGo (Silver et al., 2016). Taking the Game of Go as an example, the algorithm assists in selecting the best possible action in the current state of the board based on their average rewards.

- 例如，考虑选择动作 a1 作为当前状态(state)的下一个动作。然后游戏继续进行，所有后续行动，无论是我们这边还是对手，都由策略模型而不是真正的玩家决定。
- 整个游戏序列构成动作 a1 的一个模拟 【Simulation】。
- 对于每局游戏来说，如果我们赢了，奖励是 1;否则，奖励为 0。
- 具体来说，如果我们从当前状态的下一动作 a1 开始， 模拟 10 场比赛并赢得其中的 9 场，则 a1 的平均奖励将为 0.9。
- 然而，由于围棋游戏中的动作空间广阔，模拟每一个可能的动作是不切实际的。
- 应用于树结构的上置信界1（UCT）算法能够识别出更具获胜潜力的行动，并将更多模拟资源分配给这些行动，而非在所有行动间平均分配模拟资源。
- 一旦通过该过程确定了一个动作并实际执行，从而进入新的棋局状态，接下来便会对这个新状态应用相同的流程来选择动作，如此反复规划，直至围棋游戏实际结束。



## MCTS 的4个阶段
- 选择（Selection）
遍历当前的推理树，在推理树中选择 UCT 分数最高的子节点， 将该子节点作为模拟 (Simulation)

- 扩展（Expansion）
将上一步选中的 UCT 分数最高的节点的子节点全部加入到推理树中， 达到扩展推理树的目的。

- 模拟（Simulation）
从上一步新加入推理树中的子节点开始，继续扩展推理树，直到整个任务完成。


- 回传（Backpropagation）
在每次模拟（Simulation）之后，使用新获取的模拟奖励 （Simulation reward） 来更新推理树中所有相关节点的平均奖励（Average Reward）。


