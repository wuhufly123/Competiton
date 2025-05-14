比赛的文档看比赛库里的文档
先把比赛的环境配置好





# ubt_solution模型使用：

pip完文件底下 req后就直接可以用了，然后注意run.sh里的文件路径，得自己调对





# Universal Behavioral Transformer (UBT)

这是一个基于Transformer的用户行为建模解决方案，用于生成通用的用户行为画像。该方案通过多任务学习框架，同时优化流失预测、品类倾向性和产品倾向性等多个任务，从而生成更具表达力的用户表示。

## 主要特点

1. **多模态行为编码**：
   - 时序信息编码
   - 行为类型编码
   - 商品特征编码
   - 用户属性编码

2. **多任务学习**：
   - 流失预测
   - 品类倾向性预测
   - 产品倾向性预测

3. **高级特征提取**：
   - 注意力机制
   - 时序Transformer
   - 行为序列建模

4. **优化策略**：
   - 对比学习增强
   - 动态任务权重
   - 课程学习

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd ubt_solution
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备

确保数据目录结构如下：
```
data/
├── product_buy.parquet
├── add_to_cart.parquet
├── remove_from_cart.parquet
├── page_visit.parquet
├── search_query.parquet
├── product_properties.parquet
├── input/
│   └── relevant_clients.npy
└── target/
    ├── propensity_category.npy
    ├── propensity_sku.npy
    └── active_clients.npy
```

### 2. 训练模型

使用以下命令训练模型并生成用户嵌入向量：

cd到ubt_solution路径中：
```bash
nohup bash run.sh & disown
```

### 3. 参数说明

nohup bash run.sh & disown

### 4. 输出

模型会在指定的嵌入向量目录下生成两个文件：
- `client_ids.npy`：客户端ID数组
- `embeddings.npy`：用户嵌入向量矩阵

## 模型架构

### 1. 主要组件

- **时序编码器**：使用Transformer编码器处理用户行为序列
- **行为类型编码器**：编码不同类型的用户行为
- **商品特征编码器**：编码商品相关的特征
- **多任务预测头**：为不同任务提供预测

### 2. 训练策略

- **多任务学习**：同时优化多个预测任务
- **对比学习**：通过对比学习提高表示的区分度
- **动态权重**：根据任务性能动态调整任务权重




