import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init # Added init for _initialize_weights
from typing import Dict, List, Optional, Tuple # Though not directly used by this class, good practice if it were
from ..config import Config
import logging

logger = logging.getLogger(__name__)

class EnhancedFeatureEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 事件类型编码
        self.event_embedding = nn.Sequential(
            nn.Embedding(5, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # 商品特征编码
        self.category_embedding = nn.Sequential(
            nn.Embedding(config.num_categories + 1, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        self.price_embedding = nn.Sequential(
            nn.Embedding(101, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # 名称和查询编码
        self.name_encoder = nn.Sequential(
            nn.Linear(16, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.query_encoder = nn.Sequential(
            nn.Linear(16, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 时间特征编码
        self.time_encoder = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # 特征重要性评估 - 使用更稳定的结构
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.LayerNorm(64),  # 添加归一化层
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 特征融合 - 减小维度避免过大值 
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 6, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Tanh()  # 使用tanh限制输出范围在[-1,1]
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化方法
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # 减小初始权重方差

    def forward(self, 
                event_types: torch.Tensor,
                categories: torch.Tensor,
                prices: torch.Tensor,
                names: torch.Tensor,
                queries: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = names.shape
        
        # 确保索引在有效范围内并且没有NaN值
        event_types = torch.clamp(event_types, 0, 4)
        categories = torch.clamp(categories, 0, self.config.num_categories)
        prices = torch.clamp(prices, 0, 100)
        
        # 检查并清理NaN值
        if torch.isnan(names).any():
            names = torch.where(torch.isnan(names), torch.zeros_like(names), names)
        if torch.isnan(queries).any():
            queries = torch.where(torch.isnan(queries), torch.zeros_like(queries), queries)
        if torch.isnan(timestamps).any():
            timestamps = torch.where(torch.isnan(timestamps), torch.zeros_like(timestamps), timestamps)
        
        # 规范化时间戳以避免大数值
        timestamps = (timestamps - timestamps.mean()) / (timestamps.std() + 1e-8)
        
        # 编码事件类型
        event_emb = self.event_embedding(event_types)
        
        # 编码商品特征
        cat_emb = self.category_embedding(categories)
        price_emb = self.price_embedding(prices)
        
        # 编码名称和查询
        names_2d = names.view(-1, 16)
        name_emb_2d = self.name_encoder(names_2d)
        name_emb = name_emb_2d.view(batch_size, seq_len, self.config.hidden_size)
        
        queries_2d = queries.view(-1, 16)
        query_emb_2d = self.query_encoder(queries_2d)
        query_emb = query_emb_2d.view(batch_size, seq_len, self.config.hidden_size)
        
        # 编码时间特征
        time_emb = self.time_encoder(timestamps.unsqueeze(-1))
        
        # 计算特征重要性 - 添加数值稳定性，包含SKU特征
        features = [event_emb, cat_emb, price_emb, name_emb, query_emb, time_emb]
        importance_scores = []
        for feat in features:
            # 添加特征归一化
            norm_feat = F.normalize(feat, p=2, dim=-1)
            score = self.importance_net(norm_feat)
            importance_scores.append(score)
        
        # 加权特征融合 - 使用softmax确保权重和为1
        all_scores = torch.cat(importance_scores, dim=-1)
        normalized_scores = F.softmax(all_scores, dim=-1).unsqueeze(-1)
        
        weighted_features = []
        for i, feat in enumerate(features):
            weighted_features.append(feat * normalized_scores[:,:,i])
        
        # 特征融合
        features = torch.cat(weighted_features, dim=-1)
        fused_features = self.feature_fusion(features)
        
        # 检查并处理NaN值
        if torch.isnan(fused_features).any():
            logger.warning("NaN values detected in feature fusion output. Replacing with zeros.")
            fused_features = torch.where(
                torch.isnan(fused_features), 
                torch.zeros_like(fused_features), 
                fused_features
            )
        
        # 添加额外的梯度裁剪
        fused_features = torch.clamp(fused_features, min=-5.0, max=5.0)
        
        return fused_features 