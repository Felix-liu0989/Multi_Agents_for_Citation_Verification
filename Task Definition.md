# Task Definition

(1) 输入一个句子和支撑材料，生成初步的答案

(2) 自动评估

(3) 生成反馈

(4) 基于反馈对答案进行优化



# Algorithm 1 迭代式响应优化与指标引导反馈

伪代码

![img](https://deepseek-api-files.obs.cn-east-3.myhuaweicloud.com/raw/2025/06/10/file-700b0175-2d8f-45da-80c4-c62251fc3da1?response-content-disposition=attachment%3B+filename%3D%221749546958306.png%22&AccessKeyId=OD83TSXECLFQNNSZ3IF6&Expires=1749633949&Signature=sx4hrQqWBUXr230eYJufE5PcBcg%3D)



## 算法输入
- `M`: 基础模型(如大型语言模型)
- `x`: 输入句子/问题
- `e`: 相关证据/上下文
- `p_init`: 任务特定的初始指令提示
- `p_fb`: 任务特定的反馈提示
- `p_refine`: 任务特定的优化提示
- `K`: 最大迭代次数

## 算法步骤详解

1. **初始化阶段**:
   - 生成初始响应 `Y_0`，通过将初始提示 `p_init`、输入 `x` 和证据 `e` 拼接后输入模型 `M`
   - 数学表示为: `Y_0 = M([p_init; x; e])`

2. **迭代优化循环** (从 t=0 到 K-1):
   a. **评估当前输出**:
      - 对当前输出 `Y_t` 进行评估，得到评分 `S_t`
   
   b. **生成反馈**:
      - 将反馈提示 `p_fb`、评分 `S_t`、输入 `x`、证据 `e` 和当前输出 `Y_t` 拼接
      - 输入模型 `M` 生成反馈 `F_t`
      - 数学表示为: `F_t = M([p_fb; S_t; x; e; Y_t])`
   
   c. **基于反馈优化响应**:
      - 使用优化提示 `p_refine` 结合输入 `x`、证据 `e`、当前输出 `Y_t` 和反馈 `F_t`
      - 生成优化后的新响应 `Y_{t+1}`
      - 数学表示为: `Y_{t+1} = M([p_refine; x; e; Y_t; F_t])`
   
   d. **停止条件检查**:
      - 如果满足停止条件(如达到满意分数或收敛)，则提前终止循环

3. **最终输出**:
   - 返回优化后的输出 `Ŷ`，通过将所有历史信息 `H` (包含优化提示、输入、证据及所有迭代中的输出和反馈)输入模型 `M` 生成
   - 数学表示为: `Ŷ = M(H)`, 其中 `H = [p_refine; x; e; Y_0; F_0, ..., Y_t; F_t]`



