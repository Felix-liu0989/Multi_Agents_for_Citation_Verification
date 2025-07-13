#!/usr/bin/env python3
"""
Parquet 到 JSONL 转换工具
支持多种转换方式和错误处理
"""

import json
import os
import sys
from typing import Optional, Dict, Any, List

def check_parquet_dependencies() -> bool:
    """
    检查是否有可用的 Parquet 处理库
    """
    try:
        import pandas as pd
        # 尝试导入 pyarrow
        try:
            import pyarrow
            print("✓ pyarrow 可用")
            return True
        except ImportError:
            pass
        
        # 尝试导入 fastparquet
        try:
            import fastparquet
            print("✓ fastparquet 可用")
            return True
        except ImportError:
            pass
        
        print("✗ 未找到 pyarrow 或 fastparquet")
        return False
        
    except ImportError:
        print("✗ pandas 未安装")
        return False

def install_parquet_dependencies():
    """
    安装 Parquet 处理依赖
    """
    print("正在安装 pyarrow...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
        print("✓ pyarrow 安装成功")
        return True
    except Exception as e:
        print(f"✗ 安装失败: {e}")
        return False

def parquet_to_jsonl_pandas(parquet_file_path: str, jsonl_file_path: str, 
                           chunk_size: Optional[int] = None) -> bool:
    """
    使用 Pandas 将 Parquet 文件转换为 JSONL 文件
    
    Args:
        parquet_file_path: Parquet 文件路径
        jsonl_file_path: 输出 JSONL 文件路径
        chunk_size: 分块处理大小（用于大文件）
    
    Returns:
        bool: 转换是否成功
    """
    try:
        import pandas as pd
        
        print(f"正在读取 Parquet 文件: {parquet_file_path}")
        
        if chunk_size:
            # 分块处理大文件
            print(f"使用分块处理，块大小: {chunk_size}")
            
            # 创建输出文件
            with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                # 逐块读取和写入
                for chunk in pd.read_parquet(parquet_file_path, chunksize=chunk_size):
                    for _, row in chunk.iterrows():
                        json.dump(row.to_dict(), f, ensure_ascii=False)
                        f.write('\n')
        else:
            # 一次性处理
            print("一次性读取整个文件...")
            df = pd.read_parquet(parquet_file_path)
            print(f"读取了 {len(df)} 行数据")
            
            # 转换为 JSONL 格式
            with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                for _, row in df.iterrows():
                    json.dump(row.to_dict(), f, ensure_ascii=False)
                    f.write('\n')
        
        print(f"✓ 成功转换到: {jsonl_file_path}")
        return True
        
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ 文件未找到: {parquet_file_path}")
        return False
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return False

def parquet_to_jsonl_alternative(parquet_file_path: str, jsonl_file_path: str) -> bool:
    """
    使用替代方法转换 Parquet 文件（当 pandas 不可用时）
    """
    try:
        # 尝试使用 pyarrow 直接处理
        import pyarrow.parquet as pq
        
        print("使用 pyarrow 直接处理...")
        
        # 读取 Parquet 文件
        table = pq.read_table(parquet_file_path)
        
        # 转换为字典列表
        data = table.to_pydict()
        
        # 转换为行格式
        rows = []
        for i in range(len(next(iter(data.values())))):
            row = {key: values[i] for key, values in data.items()}
            rows.append(row)
        
        # 写入 JSONL 文件
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False, default=str)
                f.write('\n')
        
        print(f"✓ 成功转换到: {jsonl_file_path}")
        return True
        
    except ImportError:
        print("✗ pyarrow 不可用")
        return False
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return False

def convert_parquet_to_jsonl(parquet_file_path: str, jsonl_file_path: str, 
                           chunk_size: Optional[int] = None, 
                           auto_install: bool = True) -> bool:
    """
    主要的转换函数，包含完整的错误处理和依赖管理
    
    Args:
        parquet_file_path: Parquet 文件路径
        jsonl_file_path: 输出 JSONL 文件路径
        chunk_size: 分块处理大小
        auto_install: 是否自动安装依赖
    
    Returns:
        bool: 转换是否成功
    """
    print("=== Parquet 到 JSONL 转换工具 ===\n")
    
    # 检查文件是否存在
    if not os.path.exists(parquet_file_path):
        print(f"✗ 输入文件不存在: {parquet_file_path}")
        return False
    
    # 检查依赖
    if not check_parquet_dependencies():
        if auto_install:
            print("\n尝试自动安装依赖...")
            if not install_parquet_dependencies():
                print("✗ 无法安装依赖，请手动安装: pip install pyarrow")
                return False
        else:
            print("✗ 缺少必要的依赖，请安装: pip install pyarrow")
            return False
    
    # 尝试转换
    print(f"\n开始转换: {parquet_file_path} -> {jsonl_file_path}")
    
    # 方法1：使用 pandas
    if parquet_to_jsonl_pandas(parquet_file_path, jsonl_file_path, chunk_size):
        return True
    
    # 方法2：使用 pyarrow 直接处理
    if parquet_to_jsonl_alternative(parquet_file_path, jsonl_file_path):
        return True
    
    print("✗ 所有转换方法都失败了")
    return False

def main():
    """主函数，用于命令行使用"""
    if len(sys.argv) < 3:
        print("使用方法: python parquet_converter.py <parquet_file> <jsonl_file> [chunk_size]")
        print("示例: python parquet_converter.py data.parquet output.jsonl 1000")
        return
    
    parquet_file = sys.argv[1]
    jsonl_file = sys.argv[2]
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    success = convert_parquet_to_jsonl(parquet_file, jsonl_file, chunk_size)
    
    if success:
        print("\n🎉 转换完成！")
    else:
        print("\n❌ 转换失败！")
        sys.exit(1)

if __name__ == "__main__":
    main() 