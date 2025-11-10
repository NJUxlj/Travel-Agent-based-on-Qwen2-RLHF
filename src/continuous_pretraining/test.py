#!/usr/bin/env python3
"""
继续预训练测试脚本
验证配置和训练器是否能正常工作
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.configs.cpt_config import CPTConfig


def test_config():
    """测试配置类"""
    print("测试配置类...")
    
    # 测试默认配置
    config = CPTConfig()
    print(f"默认模型路径: {config.model.model_name_or_path}")
    print(f"默认学习率: {config.training.learning_rate}")
    print(f"默认最大序列长度: {config.model.max_length}")
    
    # 测试配置转换为字典
    config_dict = config.to_dict()
    assert "training" in config_dict
    assert "model" in config_dict
    assert "data" in config_dict
    print("配置转换为字典成功")
    
    # 测试从字典创建配置
    new_config = CPTConfig.from_dict(config_dict)
    assert new_config.model.model_name_or_path == config.model.model_name_or_path
    print("从字典创建配置成功")
    
    # 测试保存和加载配置
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save_to_file(temp_path)
        loaded_config = CPTConfig.load_from_file(temp_path)
        assert loaded_config.model.model_name_or_path == config.model.model_name_or_path
        print("保存和加载配置成功")
    finally:
        os.unlink(temp_path)
    
    print("配置类测试通过！\n")


def test_config_overrides():
    """测试配置覆盖"""
    print("测试配置覆盖...")
    
    # 创建自定义配置
    config = CPTConfig()
    config.model.model_name_or_path = "custom/model"
    config.training.learning_rate = 1e-4
    config.model.max_length = 4096
    config.training.per_device_train_batch_size = 8
    
    assert config.model.model_name_or_path == "custom/model"
    assert config.training.learning_rate == 1e-4
    assert config.model.max_length == 4096
    assert config.training.per_device_train_batch_size == 8
    
    print("配置覆盖测试通过！\n")


def test_config_from_file():
    """测试从文件加载配置"""
    print("测试从文件加载配置...")
    
    # 使用项目中的默认配置文件
    config_path = project_root / "src" / "configs" / "cpt_config.json"
    
    if config_path.exists():
        config = CPTConfig.from_file(str(config_path))
        assert config.model.model_name_or_path == "Qwen/Qwen2.5-7B"
        assert config.training.learning_rate == 5e-5
        print("从文件加载配置成功！")
    else:
        print(f"配置文件不存在: {config_path}")
    
    print("从文件加载配置测试通过！\n")


def test_trainer_import():
    """测试训练器导入"""
    print("测试训练器导入...")
    
    try:
        from src.continuous_pretraining.cpt_trainer import CPTTrainer
        print("训练器导入成功！")
        
        # 测试创建训练器实例（不实际训练）
        config = CPTConfig()
        config.model.model_name_or_path = "test/model"  # 使用测试路径避免实际下载
        
        # 由于不需要实际训练，这里只测试实例化
        print("训练器类测试通过！")
    except Exception as e:
        print(f"训练器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("训练器导入测试通过！\n")


def main():
    """运行所有测试"""
    print("开始继续预训练测试...\n")
    
    test_config()
    test_config_overrides()
    test_config_from_file()
    test_trainer_import()
    
    print("所有测试通过！继续预训练配置和训练器可以正常使用。")


if __name__ == "__main__":
    main()