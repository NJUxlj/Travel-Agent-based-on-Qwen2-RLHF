# 继续预训练 (Continued Pretraining) 指南

本指南介绍如何使用提供的脚本对Qwen2.5模型进行继续预训练。

## 文件结构

- `src/continuous_pretraining/cpt_trainer.py`: 继续预训练的主脚本
- `src/configs/cpt_config.py`: 配置类定义
- `src/configs/cpt_config.json`: 默认配置文件

## 使用方法

### 1. 准备训练数据

将您的训练数据准备为文本文件，每行一个文本样本：

```
这是第一个训练样本...
这是第二个训练样本...
...
```

### 2. 配置参数

您可以直接编辑 `src/configs/cpt_config.json` 文件，或者通过命令行参数覆盖配置。

### 3. 运行训练

```bash
# 基本用法
python -m src.continuous_pretraining.cpt_trainer --train_data /path/to/train_data.txt

# 使用评估数据
python -m src.continuous_pretraining.cpt_trainer \
    --train_data /path/to/train_data.txt \
    --eval_data /path/to/eval_data.txt

# 覆盖配置参数
python -m src.continuous_pretraining.cpt_trainer \
    --train_data /path/to/train_data.txt \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --output_dir ./my_cpt_output \
    --learning_rate 3e-5 \
    --batch_size 8 \
    --num_train_epochs 5 \
    --max_length 4096
```

## 配置参数说明

### 训练参数 (training)

- `learning_rate`: 学习率，默认为 5e-5
- `weight_decay`: 权重衰减，默认为 0.01
- `num_train_epochs`: 训练轮数，默认为 3
- `per_device_train_batch_size`: 每设备批大小，默认为 4
- `gradient_accumulation_steps`: 梯度累积步数，默认为 8
- `lr_scheduler_type`: 学习率调度器类型，默认为 "cosine"
- `warmup_ratio`: 预热比例，默认为 0.05
- `output_dir`: 输出目录，默认为 "./cpt_output"
- `logging_steps`: 日志记录步数，默认为 10
- `save_steps`: 保存步数，默认为 500
- `bf16`: 是否使用bfloat16，默认为 true
- `gradient_checkpointing`: 是否使用梯度检查点，默认为 true

### 模型参数 (model)

- `model_name_or_path`: 模型名称或路径，默认为 "Qwen/Qwen2.5-7B"
- `torch_dtype`: 模型数据类型，默认为 "bfloat16"
- `device_map`: 设备映射，默认为 "auto"
- `max_length`: 最大序列长度，默认为 2048
- `use_flash_attention`: 是否使用Flash Attention，默认为 true

### 数据参数 (data)

- `max_train_samples`: 最大训练样本数，默认为 None（使用全部）
- `max_eval_samples`: 最大评估样本数，默认为 None（使用全部）
- `mlm_probability`: MLM掩码概率，默认为 0.15
- `min_length`: 最小文本长度，默认为 10
- `filter_duplicates`: 是否过滤重复样本，默认为 true

## 最佳实践

1. **学习率调整**: 继续预训练通常使用较小的学习率，如 1e-5 到 5e-5 之间
2. **批大小**: 根据GPU内存调整批大小，可以通过梯度累积来增加有效批大小
3. **序列长度**: 根据数据特点调整最大序列长度，较长的序列可能需要更多GPU内存
4. **数据质量**: 确保训练数据质量，过滤低质量或重复的文本
5. **评估**: 定期在验证集上评估模型性能，避免过拟合

## 输出文件

训练过程中会生成以下文件：

- `best_model/`: 最佳模型检查点
- `final_model/`: 最终模型
- `checkpoint-epoch-{N}/`: 每个epoch的检查点
- `logs/`: 训练日志
- `cpt_config.json`: 使用的配置文件

## 故障排除

1. **内存不足**: 减小 `per_device_train_batch_size` 或启用 `gradient_checkpointing`
2. **训练不稳定**: 尝试降低学习率或调整 `warmup_ratio`
3. **收敛慢**: 增加 `num_train_epochs` 或调整学习率调度器

## 参考文献

继续预训练的最佳实践参考了以下资源：
- Hugging Face Transformers文档
- Qwen2.5模型文档
- 大语言模型继续预训练研究论文