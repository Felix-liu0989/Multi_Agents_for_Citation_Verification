#!/usr/bin/env python3
"""
简化的 Parquet 到 JSONL 转换脚本
"""

import json
import pandas as pd

def simple_parquet_to_jsonl(parquet_file: str, jsonl_file: str):
    """
    简单的 Parquet 到 JSONL 转换函数
    
    Args:
        parquet_file: Parquet 文件路径
        jsonl_file: 输出 JSONL 文件路径
    """
    try:
        # 读取 Parquet 文件
        print(f"正在读取: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        print(f"读取了 {len(df)} 行数据")
        
        # 转换为 JSONL 格式
        print(f"正在转换到: {jsonl_file}")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # 将每行转换为字典并写入
                json.dump(row.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        
        print("✓ 转换完成！")
        
    except ImportError:
        print("✗ 缺少依赖，请安装: pip install pyarrow")
    except FileNotFoundError:
        print(f"✗ 文件未找到: {parquet_file}")
    except Exception as e:
        print(f"✗ 转换失败: {e}")

# 使用示例
if __name__ == "__main__":
    # 替换为您的文件路径
    parquet_path = '0000.parquet'
    jsonl_path = 'SurveyEval.jsonl'
    
    simple_parquet_to_jsonl(parquet_path, jsonl_path) 