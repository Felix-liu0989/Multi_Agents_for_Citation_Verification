import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_iterations(file_path):
    # 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 存储迭代次数大于1的样本的分数
    scores = {
        'correct_score': [],
        'citation_recall': [],
        'citation_precision': [],
        'citation_f1': [],
        'comprehensive_score': []
    }
    
    # 统计迭代次数
    iteration_counts = defaultdict(int)
    
    for item in data:
        if 'iterations' in item and len(item['iterations']) > 1:
            # 记录迭代次数
            iteration_counts[len(item['iterations'])] += 1
            
            # 获取最后一次迭代的分数
            last_iteration = item['iterations'][-1]
            scores['correct_score'].append(last_iteration['correct_score'])
            scores['citation_recall'].append(last_iteration['evaluation']['citation_recall'])
            scores['citation_precision'].append(last_iteration['evaluation']['citation_precision'])
            scores['citation_f1'].append(last_iteration['evaluation']['citation_f1'])
            scores['comprehensive_score'].append(last_iteration['comprehensive_score'])
    
    # 打印统计信息
    print("\n迭代次数分布:")
    for count, num in sorted(iteration_counts.items()):
        print(f"迭代{count}次的样本数: {num}")
    
    print("\n分数统计:")
    for metric, values in scores.items():
        if values:  # 确保有数据
            print(f"\n{metric}:")
            print(f"平均值: {np.mean(values):.3f}")
            print(f"中位数: {np.median(values):.3f}")
            print(f"标准差: {np.std(values):.3f}")
            print(f"最小值: {min(values):.3f}")
            print(f"最大值: {max(values):.3f}")
    
    # 绘制分数分布直方图
    plt.figure(figsize=(15, 10))
    for i, (metric, values) in enumerate(scores.items(), 1):
        plt.subplot(2, 3, i)
        plt.hist(values, bins=20, alpha=0.7)
        plt.title(f'{metric} Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    plt.close()

if __name__ == "__main__":
    file_path = "preds/tmp/deepseek_chat_0616.json"
    analyze_iterations(file_path) 