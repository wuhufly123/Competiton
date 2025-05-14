from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import logging
import os
import argparse

logger = logging.getLogger(__name__)

@dataclass
class Config:
    # 模型架构参数
    hidden_size: int = 128  # 增加隐藏层大小以提升模型表达能力
    num_heads: int = 2  # 增加注意力头数量以捕捉更丰富的特征
    num_layers: int = 1  # 增加层数以提升模型深度
    dropout: float = 0.2  # 增加 dropout 以防止过拟合
    output_dim: int = 512  # 增加输出维度以容纳更多信息
    
    # 特征维度
    num_categories: int = 100
    num_products: int = 100  # 添加产品数量参数
    num_behaviors: int = 5  # 购买、加购、移除、页面访问、搜索
    name_vector_dim: int = 16
    query_vector_dim: int = 16
    
    # 行为类型配置
    max_seq_length: int = 300  # 增加序列长度以捕捉更长的行为历史
    
    # 训练配置
    batch_size: int = 8192  # 调整 batch size 以平衡训练速度和内存使用
    learning_rate: float = 5e-5  # 降低学习率以提高训练稳定性
    weight_decay: float = 1e-3  # 增加权重衰减以防止过拟合
    num_epochs: int = 150  # 增加训练轮数以充分优化模型
    warmup_steps: int = 2000  # 增加预热步数以平滑学习率变化
    patience: int = 5  # 增加早停耐心以避免过早停止
    
    # 设备配置
    accelerator: str = "cuda"
    devices: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 10
    output_dir: str = "./outputs"
    device: str = "cuda:0"
    use_cpu: bool = False  # 添加CPU选项
    
    # 多任务学习配置
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        'churn': 1.0,
        'category_propensity': 0.5,
        'product_propensity': 0.5
    })
    
    # 稳定性参数
    gradient_norm_clip: float = 1.0  # 梯度范数裁剪值
    embedding_dropout: float = 0.1  # embedding层的dropout
    attention_dropout: float = 0.1  # 注意力的dropout
    relu_dropout: float = 0.1  # ReLU激活后的dropout
    residual_dropout: float = 0.1  # 残差连接的dropout
    
    padding_idx: Dict[str, int] = field(default_factory=lambda: {
        'category': 0, 
        'price': 0,
        # Add other features here if they also need specific padding indices for embedding layers
    })
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                'churn': 1.0,
                'category_propensity': 0.8,
                'product_propensity': 0.8
            }
            
        # 检查 hidden_size 是否能被 num_heads 整除
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
            
        # 检查 CUDA 可用性
        if self.use_cpu or not torch.cuda.is_available():
            logger.warning("Using CPU for training")
            self.accelerator = "cpu"
            self.device = "cpu"
        else:
            try:
                # 尝试创建一个小的CUDA张量来测试CUDA是否正常工作
                test_tensor = torch.zeros(1, device="cuda")
                del test_tensor
                
                # 确保设备字符串格式正确
                if not self.device.startswith("cuda:"):
                    self.device = f"cuda:{self.devices[0]}"
                
                # 检查 CUDA 版本
                cuda_version = torch.version.cuda
                logger.info(f"Using CUDA version: {cuda_version}")
                
                if cuda_version < "11.0":
                    logger.warning(f"CUDA version {cuda_version} may be too old. Recommended: 11.0 or higher")
            except Exception as e:
                logger.warning(f"CUDA initialization failed: {e}. Falling back to CPU")
                self.accelerator = "cpu"
                self.device = "cpu"

# 创建默认配置实例
config = Config()

