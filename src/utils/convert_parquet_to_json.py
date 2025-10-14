#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parquet 到 JSON 格式转换工具
支持单个文件和批量转换
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from tqdm import tqdm


def convert_parquet_to_json(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    encoding: str = 'utf-8',
    indent: int = 2,
    orient: str = 'records'
) -> str:
    """
    将 Parquet 文件转换为 JSON 格式
    
    Args:
        input_path: 输入的 Parquet 文件路径
        output_path: 输出的 JSON 文件路径，如果为 None 则自动生成
        encoding: JSON 文件编码
        indent: JSON 缩进空格数
        orient: pandas to_json 的 orient 参数 ('records', 'index', 'values', 'table')
    
    Returns:
        输出文件路径
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    if not input_path.suffix.lower() == '.parquet':
        raise ValueError(f"输入文件不是 Parquet 格式: {input_path}")
    
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        output_path = input_path.with_suffix('.json')
    else:
        output_path = Path(output_path)
    
    print(f"正在读取 Parquet 文件: {input_path}")
    
    # 读取 Parquet 文件
    try:
        df = pd.read_parquet(input_path)
        print(f"成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        raise RuntimeError(f"读取 Parquet 文件失败: {e}")
    
    # 转换为 JSON
    print(f"正在转换为 JSON 格式...")
    try:
        json_str = df.to_json(orient=orient, indent=indent, force_ascii=False)
        
        # 写入文件
        with open(output_path, 'w', encoding=encoding) as f:
            f.write(json_str)
        
        print(f"转换完成！输出文件: {output_path}")
        print(f"文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(output_path)
        
    except Exception as e:
        raise RuntimeError(f"转换或保存 JSON 文件失败: {e}")


def batch_convert_parquet_to_json(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    encoding: str = 'utf-8',
    indent: int = 2,
    orient: str = 'records'
) -> list:
    """
    批量转换目录下的所有 Parquet 文件为 JSON 格式
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为 None 则在输入目录下创建 json 子目录
        encoding: JSON 文件编码
        indent: JSON 缩进空格数
        orient: pandas to_json 的 orient 参数
    
    Returns:
        转换成功的文件路径列表
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / 'json'
    
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 Parquet 文件
    parquet_files = list(input_dir.glob('*.parquet'))
    
    if not parquet_files:
        print(f"在目录 {input_dir} 中未找到 Parquet 文件")
        return []
    
    print(f"找到 {len(parquet_files)} 个 Parquet 文件")
    
    success_files = []
    
    # 批量转换
    for parquet_file in tqdm(parquet_files, desc="转换进度"):
        try:
            output_file = output_dir / f"{parquet_file.stem}.json"
            convert_parquet_to_json(
                parquet_file, 
                output_file, 
                encoding=encoding, 
                indent=indent, 
                orient=orient
            )
            success_files.append(str(output_file))
        except Exception as e:
            print(f"转换文件 {parquet_file} 失败: {e}")
    
    print(f"批量转换完成！成功转换 {len(success_files)} 个文件")
    return success_files


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='将 Parquet 文件转换为 JSON 格式')
    parser.add_argument('--input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', help='输出文件或目录路径')
    parser.add_argument('-e', '--encoding', default='utf-8', help='JSON 文件编码 (默认: utf-8)')
    parser.add_argument('-i', '--indent', type=int, default=2, help='JSON 缩进空格数 (默认: 2)')
    parser.add_argument('--orient', default='records', 
                       choices=['records', 'index', 'values', 'table'],
                       help='JSON 格式 (默认: records)')
    parser.add_argument('--batch', action='store_true', help='批量转换目录下的所有 Parquet 文件')
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            # 批量转换模式
            success_files = batch_convert_parquet_to_json(
                args.input,
                args.output,
                encoding=args.encoding,
                indent=args.indent,
                orient=args.orient
            )
            print(f"批量转换完成，共转换 {len(success_files)} 个文件")
        else:
            # 单文件转换模式
            output_path = convert_parquet_to_json(
                args.input,
                args.output,
                encoding=args.encoding,
                indent=args.indent,
                orient=args.orient
            )
            print(f"转换完成: {output_path}")
            
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    '''
    python /root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/utils/convert_parquet_to_json.py \
        --input /root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/MultiTurn_GRPO/data/train-00000-of-00001.parquet \
        -o /root/autodl-tmp/Travel-Agent-based-on-Qwen2-RLHF/src/data/MultiTurn_GRPO/json
    '''
    exit(main())
