import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Dict
from ..config import Config

class TaskSpecificEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config # Store config
        # 共享底层表示
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 任务特定编码器 - 使用更复杂的结构，便于学习更好的特征表示
        self.task_encoders = nn.ModuleDict({
            'churn': nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Tanh()  # 限制输出范围
            ),
            'category_propensity': nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),  # 更大的网络容量
                nn.LayerNorm(config.hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Tanh()  # 限制输出范围
            ),
            'product_propensity': nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),  # 更大的网络容量
                nn.LayerNorm(config.hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.Tanh()  # 限制输出范围
            )
        })
        
        # 任务头 - 不再使用注意力机制，简化模型结构避免维度问题
        self.category_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.product_projector = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.task_heads = nn.ModuleDict({
            'churn': nn.Linear(config.hidden_size, 1),
            'category_propensity': nn.Linear(config.hidden_size, config.num_categories),
            'product_propensity': nn.Linear(config.hidden_size, config.num_products)
        })
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, module in self.task_heads.items():
            if isinstance(module, nn.Linear):
                # 使用较小的初始权重以避免大输出
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 特别初始化类别和产品投影器
        for module in [self.category_projector, self.product_projector]:
            nn.init.orthogonal_(module.weight)  # 使用正交初始化提高特征区分度
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, user_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        # 共享表示
        shared_features = self.shared_encoder(user_embedding)
        
        # 流失预测
        churn_features = self.task_encoders['churn'](shared_features)
        outputs['churn'] = self.task_heads['churn'](churn_features).squeeze(-1)
        
        # 类别倾向性预测 - 使用投影但避免维度问题
        category_features = self.task_encoders['category_propensity'](shared_features)
        # 使用投影器强化特征表示
        category_features = self.category_projector(category_features)
        outputs['category_propensity'] = self.task_heads['category_propensity'](category_features)
        
        # 产品倾向性预测 - 使用投影但避免维度问题
        product_features = self.task_encoders['product_propensity'](shared_features)
        # 使用投影器强化特征表示
        product_features = self.product_projector(product_features)
        outputs['product_propensity'] = self.task_heads['product_propensity'](product_features)
        
        # 裁剪输出值以增加数值稳定性
        for key in outputs:
            outputs[key] = torch.clamp(outputs[key], min=-10.0, max=10.0)
        
        return outputs 