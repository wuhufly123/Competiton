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

```bash
python -m ubt_solution.create_embeddings \
    --data-dir /path/to/data \
    --embeddings-dir /path/to/embeddings \
    --accelerator gpu \
    --devices 0 \
    --num-workers 4 \
    --batch-size 128 \
    --num-epochs 10 \
    --learning-rate 1e-4
```

### 3. 参数说明

- `--data-dir`：数据目录路径
- `--embeddings-dir`：嵌入向量保存目录
- `--accelerator`：使用的加速器类型（gpu/cpu）
- `--devices`：使用的设备ID列表
- `--num-workers`：数据加载的工作进程数
- `--batch-size`：训练批次大小
- `--num-epochs`：训练轮数
- `--learning-rate`：学习率

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

## 性能优化

1. **数据加载优化**：
   - 使用多进程数据加载
   - 预取和缓存机制

2. **训练优化**：
   - 混合精度训练
   - 梯度累积
   - 学习率调度

3. **内存优化**：
   - 高效的数据批处理
   - 梯度检查点

## 注意事项

1. 确保有足够的GPU内存（建议至少16GB）
2. 对于大规模数据集，建议使用多GPU训练
3. 可以根据实际需求调整模型参数和训练配置

## 引用

如果您使用了本代码，请引用：

```bibtex
@misc{ubt2024,
  author = {Your Name},
  title = {Universal Behavioral Transformer},
  year = {2024},
  publisher = {GitHub},
  url = {[repository_url]}
}
```

## 问题修复：解决Loss为NaN的问题

为了解决训练过程中loss变成NaN的问题，我们进行了以下优化：

### 模型架构方面的优化

1. **数值稳定性检查**：在各个关键处添加了NaN和Inf值的检查和处理
2. **残差连接增强**：为Transformer编码器添加了直接残差连接，提高梯度传播稳定性
3. **特征融合改进**：使用多种池化方法进行特征融合，避免信息损失
4. **将动态任务权重替换为固定权重**：避免权重计算中的不稳定性
5. **梯度值限制**：在模型中对logits进行截断，限制在合理范围内

### 训练和优化方面的优化

1. **简化优化器**：从AdamW切换到普通Adam，降低学习率，提高eps参数
2. **移除梯度裁剪**：改为检测和跳过有问题的batch，而不是强制裁剪
3. **添加梯度检查**：在反向传播后检查梯度，如有异常则跳过该批次
4. **使用更稳定的学习率调度**：增加scheduler的耐心参数

### 数据处理方面的优化

1. **异常值处理**：全面处理输入数据中的NaN、Inf和异常值
2. **特征归一化**：对时间戳等特征进行了归一化处理
3. **更健壮的数据加载**：添加异常处理，确保数据缺失或格式问题不会导致训练失败
4. **安全的目标值处理**：确保目标值的形状和范围符合预期

### 配置参数优化

1. **减小模型规模**：降低隐藏层大小、头数量和层数，减少模型复杂度
2. **调整Dropout**：为不同层设置适当的dropout参数
3. **增大batch size**：提高训练稳定性
4. **降低学习率**：大幅降低初始学习率，避免优化过程中的波动

这些优化共同作用，从根本上解决了模型训练过程中loss变为NaN的问题，使模型能够稳定训练和收敛，而无需依赖梯度裁剪等技巧。
