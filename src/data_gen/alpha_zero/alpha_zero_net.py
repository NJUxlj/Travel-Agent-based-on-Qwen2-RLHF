import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    残差块（Residual Block）是ResNet（残差神经网络）的核心组件
    
    ResNet解决了深层网络训练中的梯度消失问题：
    - 传统的深层网络随着层数增加，训练变得更加困难
    - 残差连接（skip connection）允许梯度直接流过网络
    - 公式：y = F(x) + x，其中F(x)是学习到的残差，x是输入
    
    在计算机视觉中，卷积神经网络（CNN）通过卷积层提取图像特征：
    - 卷积层：用卷积核在输入上滑动，提取局部特征
    - 3x3卷积核：一次处理3x3像素区域
    - padding=1：保持输入尺寸不变
    """
    def __init__(self, channels):
        super().__init__()
        
        # 第一个卷积层：提取特征
        # Conv2d(输入通道数, 输出通道数, 卷积核大小, padding, 是否使用偏置)
        # 
        # 什么是通道（Channels）？
        # - 通道可以理解为图像的"层数"或"特征图的数量"
        # - 在RGB图像中，有3个通道：红色(R)、绿色(G)、蓝色(B)
        # - 在围棋中，通常有3个通道：
        #   * 通道1：黑子位置 (1表示有黑子，0表示没有)
        #   * 通道2：白子位置 (1表示有白子，0表示没有)  
        #   * 通道3：空位位置 (1表示空位，0表示有子)
        # - 卷积层的输出通道数 = 卷积核的数量 = 提取的特征类型数量
        # - 更多的通道意味着能提取更多不同类型的特征
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # 批归一化（Batch Normalization）：加速训练，稳定梯度
        # bn1 需要知道通道数才能对每个通道分别进行归一化
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 第二个卷积层：进一步处理特征
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        """
        残差块的前向传播
        
        参数:
        x: 输入张量，形状为 (batch_size, channels, height, width)
        
        返回:
        输出张量，形状与输入相同
        
        形状变化详细说明（以围棋为例，channels=256, height=width=19）：
        输入：     x.shape = (batch, 256, 19, 19)
        ↓ 第一个卷积层 + 批归一化 + ReLU
        conv1:    (batch, 256, 19, 19) → (batch, 256, 19, 19)  [3x3卷积保持尺寸]
        bn1:      (batch, 256, 19, 19) → (batch, 256, 19, 19)  [批归一化不改变形状]
        relu:     (batch, 256, 19, 19) → (batch, 256, 19, 19)  [激活函数不改变形状]
        ↓ 第二个卷积层 + 批归一化
        conv2:    (batch, 256, 19, 19) → (batch, 256, 19, 19)  [3x3卷积保持尺寸]
        bn2:      (batch, 256, 19, 19) → (batch, 256, 19, 19)  [批归一化不改变形状]
        ↓ 残差连接 + ReLU
        x += residual: (batch, 256, 19, 19) + (batch, 256, 19, 19) = (batch, 256, 19, 19)
        relu:       (batch, 256, 19, 19) → (batch, 256, 19, 19)
        """
        # 保存原始输入（残差连接）
        residual = x
        
        # 第一个卷积层 + 批归一化 + ReLU激活
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 第二个卷积层 + 批归一化
        x = self.bn2(self.conv2(x))
        
        # 残差连接：将处理后的特征加回原始输入
        # 这是ResNet的核心：x = F(x) + x
        x += residual
        
        # 最后的ReLU激活
        return F.relu(x)

class AlphaZeroNet(nn.Module):
    """
    Alpha Zero网络：一个结合了策略和价值输出的神经网络
    
    AlphaZero使用深度学习来学习围棋等游戏的策略：
    - 输入：游戏棋盘状态（如围棋的19x19棋盘）
    - 输出：策略（每个动作的概率）+ 价值（当前局面的胜率）
    
    网络结构：
    1. 初始卷积层：提取棋盘的基本特征
    2. 残差层（多个ResidualBlock）：深度特征提取
    3. 策略头：输出每个可能动作的概率分布
    4. 价值头：评估当前局面的价值（胜负概率）
    """
    def __init__(self, game, num_channels=256, num_res_blocks=19):
        super().__init__()
        
        # 保存游戏相关参数
        self.game = game
        # 获取棋盘大小（如围棋是19x19）
        self.board_x, self.board_y = game.getBoardSize()
        # 获取动作空间大小（如围棋有361个可能位置）
        self.action_size = game.getActionSize()
        
        # =============== 初始卷积层 ===============
        # 输入3个通道（如围棋的：黑子、白子、空位），输出256个特征通道
        self.conv_initial = nn.Conv2d(3, num_channels, 3, padding=1, bias=False)
        # 批归一化：加速训练过程
        self.bn_initial = nn.BatchNorm2d(num_channels)
        
        # =============== 残差层 ===============
        # 创建19个残差块，形成深度网络
        # 残差连接使得网络可以训练得更深，避免梯度消失
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # =============== 策略头（Policy Head） ===============
        # 作用：输出每个动作的概率分布
        # 在围棋中，策略头输出361个概率值，表示在每个位置落子的概率
        
        # 1x1卷积：减少通道数到2
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        
        # 全连接层：将特征映射到动作空间
        # 输入：2 * board_x * board_y（展平后的特征图）
        # 输出：action_size（可能的动作数量）
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)
        
        # =============== 价值头（Value Head） ===============
        # 作用：评估当前局面的价值（当前玩家获胜的概率）
        # 输出值在-1到1之间：-1表示必败，1表示必胜
        
        # 1x1卷积：减少通道数到1
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        
        # 第一个全连接层：降维到256
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, 256)
        
        # 第二个全连接层：输出最终的价值估计
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        """
        前向传播：从输入棋盘状态到输出策略和价值
        
        参数:
        x: 输入张量，形状为 (batch_size, 3, board_x, board_y)
           - batch_size: 批次大小
           - 3: 通道数（围棋：黑子、白子、空位）
           - board_x, board_y: 棋盘尺寸（通常是19x19）
        
        返回:
        policy: 策略输出，形状为 (batch_size, action_size)
                每个动作的对数概率
        value: 价值输出，形状为 (batch_size, 1)
               当前局面的价值估计（-1到1之间）
        
        完整的形状变化过程（以围棋为例，batch_size=8, board_x=board_y=19）：
        
        输入层：
        输入：x.shape = (8, 3, 19, 19)
        
        初始卷积层：
        conv_initial:   (8, 3, 19, 19) → (8, 256, 19, 19)  [3x3卷积，输出256个特征通道]
        bn_initial:     (8, 256, 19, 19) → (8, 256, 19, 19)  [批归一化不改变形状]
        ReLU:           (8, 256, 19, 19) → (8, 256, 19, 19)  [激活函数不改变形状]
        
        残差层（19个残差块，每个都保持形状不变）：
        for循环:        (8, 256, 19, 19) → (8, 256, 19, 19)  [19次重复，形状不变]
        
        策略头分支：
        policy_conv:    (8, 256, 19, 19) → (8, 2, 19, 19)    [1x1卷积压缩到2通道]
        policy_bn:      (8, 2, 19, 19) → (8, 2, 19, 19)      [批归一化]
        ReLU:           (8, 2, 19, 19) → (8, 2, 19, 19)      [激活函数]
        view展平:       (8, 2, 19, 19) → (8, 722)            [展平为 (8, 2*19*19)]
        policy_fc:      (8, 722) → (8, 361)                  [全连接到361个动作]
        log_softmax:    (8, 361) → (8, 361)                  [输出对数概率]
        
        价值头分支：
        value_conv:     (8, 256, 19, 19) → (8, 1, 19, 19)    [1x1卷积压缩到1通道]
        value_bn:       (8, 1, 19, 19) → (8, 1, 19, 19)      [批归一化]
        ReLU:           (8, 1, 19, 19) → (8, 1, 19, 19)      [激活函数]
        view展平:       (8, 1, 19, 19) → (8, 361)            [展平为 (8, 19*19)]
        value_fc1:      (8, 361) → (8, 256)                  [全连接层降维]
        ReLU:           (8, 256) → (8, 256)                  [激活函数]
        value_fc2:      (8, 256) → (8, 1)                    [输出最终价值]
        tanh:           (8, 1) → (8, 1)                      [限制在[-1,1]范围]
        
        最终返回：
        policy: (8, 361)  - 每个batch中8个局面的策略概率分布
        value:  (8, 1)    - 每个batch中8个局面的价值评估
        """
        # =============== 第一步：初始卷积层 ===============
        # 输入：原始棋盘状态（3个通道）
        # 处理：3x3卷积提取基本特征 + 批归一化 + ReLU激活
        # 输出：256个特征通道的棋盘表示
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        
        # =============== 第二步：残差层 ===============
        # 通过19个残差块逐步提取深度特征
        # 每个残差块都会保持棋盘的空间尺寸（19x19）
        # 残差连接确保梯度能够流过深层网络
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # =============== 第三步：策略头（Policy Head） ===============
        # 作用：从深度特征中预测每个动作的概率
        
        # 1x1卷积：减少通道数到2（进一步压缩信息）
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        
        # 展平：将2D特征图展平为1D向量
        # 从形状 (batch, 2, 19, 19) 变为 (batch, 2*19*19)
        policy = policy.view(-1, 2 * self.board_x * self.board_y)
        
        # 全连接层：映射到动作空间
        # 输入：2*19*19 = 722维特征
        # 输出：action_size（如围棋的361个可能落子位置）
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # =============== 第四步：价值头（Value Head） ===============
        # 作用：评估当前局面的价值（当前玩家的获胜概率）
        
        # 1x1卷积：减少通道数到1（提取最重要的价值信息）
        value = F.relu(self.value_bn(self.value_conv(x)))
        
        # 展平：将1D特征图展平为向量
        # 从形状 (batch, 1, 19, 19) 变为 (batch, 19*19)
        value = value.view(-1, self.board_x * self.board_y)
        
        # 第一个全连接层：进一步处理价值特征
        value = F.relu(self.value_fc1(value))
        
        # 第二个全连接层：输出最终的价值估计
        # 使用tanh激活函数将输出限制在[-1, 1]范围内
        # -1表示当前玩家必败，1表示当前玩家必胜，0表示平局
        value = torch.tanh(self.value_fc2(value))
        
        # 返回策略和价值
        return policy, value