# Qwen2.5 继续预训练指南

## 概述

本指南介绍如何使用完善的 `CPTTrainer` 类对 Qwen2.5 模型进行高效的继续预训练。该实现集成了 Muon 优化器和多种效率优化技术。

## 主要特性

### 🚀 效率优化
- **Muon 优化器**: 相比 AdamW 提升近一倍的计算效率
- **Flash Attention**: 加速注意力计算，减少内存使用
- **梯度检查点**: 节省显存，支持更大模型
- **混合精度训练**: 使用 bfloat16 提升训练速度
- **梯度累积**: 支持大批量训练

### 📊 预训练效率提升
- **动态学习率调度**: 基于模型参数量自动调整学习率
- **批量大小优化**: 根据数据规模智能调整批量大小
- **内存优化**: 多种技术减少显存占用
- **并行数据加载**: 多进程数据预处理

## 安装依赖

```bash
# 基础依赖
pip install -r requirements_cpt.txt

# 安装 Muon 优化器 (推荐)
pip install muon-optimizer

# 安装 Flash Attention (可选，需要 CUDA)
pip install flash-attn --no-build-isolation
```

## 使用方法

### 基础使用

```python
from src.pretrain.cpt import CPTTrainer, CPTConfig

# 创建配置
config = CPTConfig(
    model_name_or_path="Qwen/Qwen2.5-7B",
    max_length=2048,
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    use_muon_optimizer=True,  # 使用 Muon 优化器
    use_flash_attention=True,  # 使用 Flash Attention
    use_gradient_checkpointing=True,  # 启用梯度检查点
    use_bf16=True,  # 使用混合精度
    output_dir="./cpt_output",
    max_samples=1000
)

# 创建训练器
trainer = CPTTrainer(config)

# 准备数据
train_texts = [
    "你的训练文本1",
    "你的训练文本2",
    # ... 更多文本
]

# 开始训练
final_loss = trainer.train(train_texts)
```

### 高级配置

```python
# 针对大规模训练的配置
config = CPTConfig(
    model_name_or_path="Qwen/Qwen2.5-7B",
    max_length=4096,  # 更长的序列长度
    learning_rate=3e-5,  # 根据模型规模调整
    num_train_epochs=5,
    per_device_train_batch_size=1,  # 小批量以节省显存
    gradient_accumulation_steps=16,  # 通过梯度累积实现大批量
    warmup_ratio=0.1,  # 更长的预热期
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # 效率优化
    use_muon_optimizer=True,
    use_flash_attention=True,
    use_gradient_checkpointing=True,
    use_bf16=True,
    dataloader_num_workers=8,  # 更多数据加载进程
    
    # 输出配置
    output_dir="./large_scale_cpt",
    logging_steps=50,
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=5,
    
    # 数据配置
    max_samples=100000  # 限制数据量
)
```

## 配置参数说明

### 模型配置
- `model_name_or_path`: 预训练模型路径
- `max_length`: 最大序列长度
- `trust_remote_code`: 是否信任远程代码

### 训练配置
- `learning_rate`: 学习率 (推荐: 3e-5 到 5e-5)
- `num_train_epochs`: 训练轮数
- `per_device_train_batch_size`: 每设备批量大小
- `gradient_accumulation_steps`: 梯度累积步数
- `warmup_ratio`: 预热比例
- `weight_decay`: 权重衰减
- `max_grad_norm`: 梯度裁剪阈值

### 效率优化配置
- `use_muon_optimizer`: 是否使用 Muon 优化器
- `use_flash_attention`: 是否使用 Flash Attention
- `use_gradient_checkpointing`: 是否使用梯度检查点
- `use_bf16`: 是否使用 bfloat16 精度
- `dataloader_num_workers`: 数据加载进程数

## 性能优化建议

### 1. 硬件要求
- **GPU**: 建议使用 24GB+ 显存的 GPU (如 A100, RTX 4090)
- **内存**: 建议 32GB+ 系统内存
- **存储**: 足够的磁盘空间存储模型和检查点

### 2. 超参数调优
```python
# 根据模型规模调整学习率
# 学习率 ∝ N^(-0.25)，其中 N 是模型参数量
if model_size == "7B":
    learning_rate = 5e-5
elif model_size == "14B":
    learning_rate = 3e-5
elif model_size == "32B":
    learning_rate = 2e-5

# 根据数据规模调整批量大小
# 批量大小 ∝ (N * D)^0.4，其中 N 是参数量，D 是数据量
effective_batch_size = (model_params * data_size) ** 0.4
```

### 3. 内存优化
```python
# 启用所有内存优化选项
config = CPTConfig(
    use_gradient_checkpointing=True,  # 节省显存
    use_bf16=True,  # 混合精度
    per_device_train_batch_size=1,  # 小批量
    gradient_accumulation_steps=16,  # 通过累积实现大批量
)
```

## 监控和调试

### 1. 日志监控
训练过程中会输出详细的日志信息，包括：
- 每个步骤的损失值
- 平均损失
- 学习率变化
- 内存使用情况

### 2. 模型保存
- 每个 epoch 后保存检查点
- 最佳模型自动保存
- 最终模型保存
- 配置文件保存

### 3. 常见问题
1. **显存不足**: 减少批量大小，增加梯度累积步数
2. **训练速度慢**: 确保启用了 Flash Attention 和 Muon 优化器
3. **收敛慢**: 调整学习率和预热比例

## 实验跟踪

可以使用 Weights & Biases 或 TensorBoard 进行实验跟踪：

```python
# 在配置中添加
config.wandb_project = "qwen2.5-cpt"
config.logging_dir = "./logs"
```

## 最佳实践

1. **数据质量**: 确保训练数据的高质量和多样性
2. **渐进式训练**: 先在小数据集上测试，再扩展到大规模数据
3. **定期评估**: 使用验证集监控模型性能
4. **检查点管理**: 定期保存检查点，避免训练中断
5. **资源监控**: 监控 GPU 和内存使用情况

## 性能基准

在标准硬件配置下的预期性能：

| 模型规模 | GPU | 批量大小 | 训练速度 | 显存使用 |
|---------|-----|---------|---------|---------|
| 7B      | A100 40GB | 4 | ~100 tokens/s | ~35GB |
| 14B     | A100 40GB | 2 | ~60 tokens/s | ~38GB |
| 32B     | A100 80GB | 1 | ~30 tokens/s | ~70GB |

*注：实际性能可能因硬件配置和数据特征而有所不同*
