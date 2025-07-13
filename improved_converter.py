#!/usr/bin/env python3
"""
改进的 Parquet 到 JSONL 转换脚本
修复了原始代码中的问题
"""

import pandas as pd
import json

def parquet_to_jsonl_improved(parquet_file_path, jsonl_file_path):
    """
    改进的 Parquet 到 JSONL 转换函数
    
    Args:
        parquet_file_path (str): Parquet 文件的路径
        jsonl_file_path (str): 输出 JSONL 文件的路径
    """
    try:
        # 读取 Parquet 文件到 Pandas DataFrame
        print(f"正在读取 Parquet 文件: {parquet_file_path}")
        df = pd.read_parquet(parquet_file_path)
        print(f"成功读取 {len(df)} 行数据")

        # 将 DataFrame 转换为 JSONL 格式
        print(f"正在转换为 JSONL 格式: {jsonl_file_path}")
        
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # 将每行转换为字典
                row_dict = row.to_dict()
                
                # 处理 NaN 值和其他不可序列化的类型
                cleaned_dict = {}
                for key, value in row_dict.items():
                    if pd.isna(value):
                        cleaned_dict[key] = None
                    elif isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
                        cleaned_dict[key] = str(value)
                    else:
                        cleaned_dict[key] = value
                
                # 写入 JSONL 格式（每行一个 JSON 对象）
                json.dump(cleaned_dict, f, ensure_ascii=False, default=str)
                f.write('\n')
        
        print(f"✓ 成功将 '{parquet_file_path}' 转换为 '{jsonl_file_path}'")
        print(f"  转换了 {len(df)} 行数据")

    except ImportError as e:
        print(f"✗ 缺少必要的依赖: {e}")
        print("请安装: pip install pyarrow")
    except FileNotFoundError:
        print(f"✗ 错误：文件 '{parquet_file_path}' 未找到。")
    except Exception as e:
        print(f"✗ 转换过程中发生错误：{e}")

def parquet_to_jsonl_with_chunks(parquet_file_path, jsonl_file_path, chunk_size=1000):
    """
    分块处理大文件的 Parquet 到 JSONL 转换函数
    
    Args:
        parquet_file_path (str): Parquet 文件的路径
        jsonl_file_path (str): 输出 JSONL 文件的路径
        chunk_size (int): 每次处理的行数
    """
    try:
        print(f"正在分块读取 Parquet 文件: {parquet_file_path}")
        print(f"块大小: {chunk_size}")
        
        total_rows = 0
        
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            # 分块读取
            for chunk in pd.read_parquet(parquet_file_path, chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    # 处理每行数据
                    row_dict = row.to_dict()
                    
                    # 清理数据
                    cleaned_dict = {}
                    for key, value in row_dict.items():
                        if pd.isna(value):
                            cleaned_dict[key] = None
                        elif isinstance(value, (pd.Timestamp, pd.DatetimeTZDtype)):
                            cleaned_dict[key] = str(value)
                        else:
                            cleaned_dict[key] = value
                    
                    # 写入 JSONL 格式
                    json.dump(cleaned_dict, f, ensure_ascii=False, default=str)
                    f.write('\n')
                
                total_rows += len(chunk)
                print(f"已处理 {total_rows} 行...")
        
        print(f"✓ 成功将 '{parquet_file_path}' 转换为 '{jsonl_file_path}'")
        print(f"  总共转换了 {total_rows} 行数据")

    except ImportError as e:
        print(f"✗ 缺少必要的依赖: {e}")
        print("请安装: pip install pyarrow")
    except FileNotFoundError:
        print(f"✗ 错误：文件 '{parquet_file_path}' 未找到。")
    except Exception as e:
        print(f"✗ 转换过程中发生错误：{e}")

# 使用示例
if __name__ == "__main__":
    # 文件路径
    parquet_path = '0000.parquet'
    jsonl_path = 'SurveyEval.jsonl'
    
    print("=== Parquet 到 JSONL 转换工具 ===\n")
    
    # 方法1：一次性转换（适合小文件）
    print("方法1：一次性转换")
    parquet_to_jsonl_improved(parquet_path, jsonl_path)
    
    print("\n" + "="*50 + "\n")
    
    # 方法2：分块转换（适合大文件）
    print("方法2：分块转换")
    parquet_to_jsonl_with_chunks(parquet_path, jsonl_path.replace('.jsonl', '_chunked.jsonl'), chunk_size=1000) 