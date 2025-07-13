#!/usr/bin/env python3
"""
测试脚本：验证pred_scholar.py中的修复是否成功
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入是否成功"""
    try:
        print("测试导入...")
        
        # 测试GeminiClient导入
        from citegeist.utils.llm_clients import GeminiClient
        print("✓ GeminiClient导入成功")
        
        # 测试其他必要的导入
        from citegeist.utils.prompts import generate_win_rate_evaluation_prompt
        print("✓ generate_win_rate_evaluation_prompt导入成功")
        
        from citegeist.utils.helpers import load_api_key
        print("✓ load_api_key导入成功")
        
        return True
        
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_client_initialization():
    """测试客户端初始化"""
    try:
        print("\n测试客户端初始化...")
        
        from citegeist.utils.llm_clients import GeminiClient
        
        # 检查环境变量
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("⚠ 警告: OPENROUTER_API_KEY环境变量未设置")
            return False
        
        # 初始化客户端
        client = GeminiClient(
            api_key=api_key,
            model_name="google/gemini-2.5-flash-preview-05-20",
        )
        print("✓ GeminiClient初始化成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 客户端初始化失败: {e}")
        return False

def test_prompt_generation():
    """测试提示词生成"""
    try:
        print("\n测试提示词生成...")
        
        from citegeist.utils.prompts import generate_win_rate_evaluation_prompt
        
        # 测试提示词生成
        prompt, order = generate_win_rate_evaluation_prompt(
            source_abstract="This is a test abstract",
            source_related_work="This is source related work",
            target_related_work="This is target related work"
        )
        
        print("✓ 提示词生成成功")
        print(f"  顺序: {order}")
        print(f"  提示词长度: {len(prompt)} 字符")
        
        return True
        
    except Exception as e:
        print(f"✗ 提示词生成失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 测试pred_scholar.py修复 ===\n")
    
    # 运行所有测试
    tests = [
        test_imports,
        test_client_initialization,
        test_prompt_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有测试通过！修复成功。")
        return True
    else:
        print("✗ 部分测试失败，需要进一步检查。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 