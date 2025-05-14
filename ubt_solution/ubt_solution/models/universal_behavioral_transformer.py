import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from ..config import Config
from .enhanced_feature_encoder import EnhancedFeatureEncoder
from .sequence_encoder import SequenceEncoder
from .task_specific_encoder import TaskSpecificEncoder
import logging

logger = logging.getLogger(__name__)

class UniversalBehavioralTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.feature_encoder = EnhancedFeatureEncoder(config)
        
        self.event_specific_encoders = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ) for i in range(5)  # 假设有5种事件类型
        })
        
        self.event_transformers = nn.ModuleDict({
            str(i): SequenceEncoder(config) for i in range(5)
        })
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 5, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Tanh()
        )
        
        self.task_encoder = TaskSpecificEncoder(config)
        
        self.register_buffer('task_weights', torch.tensor([
            config.task_weights['churn'],
            config.task_weights['category_propensity'],
            config.task_weights['product_propensity']
        ]))
        
        self.loss_scale = config.loss_scale if hasattr(config, 'loss_scale') else 0.1
        self.pos_weight = config.pos_weight if hasattr(config, 'pos_weight') else 5.0
        self.use_dynamic_task_weights = config.use_dynamic_task_weights if hasattr(config, 'use_dynamic_task_weights') else False

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        feature_embeddings = self.feature_encoder(
            event_types=batch['event_types'],
            categories=batch['categories'],
            prices=batch['prices'],
            names=batch['names'],
            queries=batch['queries'],
            timestamps=batch['timestamps']
        )
        
        batch_size, seq_len, hidden_size = feature_embeddings.shape
        event_specific_emb = torch.zeros_like(feature_embeddings)
        for event_type_val in range(5):
            type_mask = (batch['event_types'] == event_type_val).unsqueeze(-1)
            event_emb = self.event_specific_encoders[str(event_type_val)](feature_embeddings)
            event_specific_emb = event_specific_emb + event_emb * type_mask
        feature_embeddings = event_specific_emb
        
        mask = batch['mask']
        
        event_user_embeddings = [torch.zeros(batch_size, hidden_size, device=device) for _ in range(5)]
        event_temporal_features = torch.zeros(batch_size, seq_len, hidden_size, device=device)

        for event_type_val in range(5):
            event_mask = (batch['event_types'] == event_type_val)
            if event_mask.sum() > 0:
                event_indices = torch.where(event_mask.any(dim=1))[0]
                if event_indices.numel() == 0: continue # Skip if no users for this event type

                valid_seq_lengths = event_mask[event_indices].sum(dim=1)
                max_seq_len_local = valid_seq_lengths.max().item()
                if max_seq_len_local == 0: continue # Skip if max_seq_len is 0

                filtered_features = torch.zeros(event_indices.size(0), max_seq_len_local, hidden_size, device=device)
                filtered_mask_for_transformer = torch.zeros(event_indices.size(0), max_seq_len_local, dtype=torch.bool, device=device)
                
                for i, user_idx in enumerate(event_indices):
                    user_event_mask = event_mask[user_idx]
                    valid_events = feature_embeddings[user_idx][user_event_mask]
                    
                    current_len = valid_events.size(0)
                    len_to_copy = min(current_len, max_seq_len_local)
                    
                    if len_to_copy > 0:
                        filtered_features[i, :len_to_copy] = valid_events[:len_to_copy]
                        if mask is not None:
                            user_original_mask = mask[user_idx][user_event_mask]
                            filtered_mask_for_transformer[i, :len_to_copy] = user_original_mask[:len_to_copy]
                        else:
                            filtered_mask_for_transformer[i, :len_to_copy] = True
                
                if filtered_features.size(0) > 0: # Ensure there are features to process
                    event_emb_output, event_temp_output = self.event_transformers[str(event_type_val)](
                        filtered_features,
                        filtered_mask_for_transformer if mask is not None else None
                    )
                    event_user_embeddings[event_type_val][event_indices] = event_emb_output
                    
                    for i, user_idx in enumerate(event_indices):
                        len_to_copy_temp = min(event_temp_output.size(1), seq_len) # event_temp_output is [N, S_local, H]
                        if len_to_copy_temp > 0 and i < event_temp_output.size(0):
                             event_temporal_features[user_idx, :len_to_copy_temp] += event_temp_output[i, :len_to_copy_temp]

        user_embeddings = self.fusion_layer(torch.cat(event_user_embeddings, dim=-1))
        temporal_features = event_temporal_features
        
        task_outputs = self.task_encoder(user_embeddings)
        
        losses = {}
        
        churn_loss = F.binary_cross_entropy_with_logits(
            task_outputs['churn'].squeeze(), batch['churn'].float(), pos_weight=torch.tensor(self.pos_weight, device=device) if batch['churn'].float().sum() > 0 else None
        )
        losses['churn_loss'] = churn_loss
        
        category_propensity_loss = F.binary_cross_entropy_with_logits(
            task_outputs['category_propensity'], 
            (batch['category_propensity'] > 0).float()
        )
        losses['category_propensity_loss'] = category_propensity_loss

        product_propensity_loss = F.binary_cross_entropy_with_logits(
            task_outputs['product_propensity'], 
            (batch['product_propensity'] > 0).float()
        )
        losses['product_propensity_loss'] = product_propensity_loss
        
        batch_size_val = batch['client_id'].size(0)
        cat_has_label = 0
        prod_has_label = 0
        if 'category_propensity' in batch:
            cat_has_label = torch.sum(torch.sum(batch['category_propensity'], dim=1) > 0).item()
            if batch_size_val > 0 : logger.info(f"品类倾向性: {cat_has_label}/{batch_size_val} 样本有标签 ({cat_has_label/batch_size_val*100:.2f}%)")
        
        if 'product_propensity' in batch:
            prod_has_label = torch.sum(torch.sum(batch['product_propensity'], dim=1) > 0).item()
            if batch_size_val > 0 : logger.info(f"产品倾向性: {prod_has_label}/{batch_size_val} 样本有标签 ({prod_has_label/batch_size_val*100:.2f}%)")
            
        if self.use_dynamic_task_weights and batch_size_val > 0:
            cat_weight_val = max(0.3, min(0.5, cat_has_label / (batch_size_val + 1e-8)))
            prod_weight_val = max(0.3, min(0.5, prod_has_label / (batch_size_val + 1e-8)))
            
            total_dynamic_weight = 1.0 + cat_weight_val + prod_weight_val
            churn_w = 1.0 / total_dynamic_weight
            cat_w = cat_weight_val / total_dynamic_weight
            prod_w = prod_weight_val / total_dynamic_weight
            
            total_loss = (
                churn_w * churn_loss +
                cat_w * category_propensity_loss +
                prod_w * product_propensity_loss
            )
            logger.info(f"动态任务权重: 流失={churn_w:.2f}, 品类={cat_w:.2f}, 产品={prod_w:.2f}")
        else:
            total_loss = (
                self.task_weights[0] * churn_loss +
                self.task_weights[1] * category_propensity_loss +
                self.task_weights[2] * product_propensity_loss
            )
            logger.info(f"固定任务权重: 流失={self.task_weights[0]:.2f}, 品类={self.task_weights[1]:.2f}, 产品={self.task_weights[2]:.2f}")

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN or Inf loss detected! Setting to a large but finite value.")
            total_loss = torch.tensor(100.0, device=device, dtype=total_loss.dtype)
        
        losses['loss'] = total_loss * self.loss_scale
        
        return {
            'user_embedding': user_embeddings,
            'temporal_features': temporal_features,
            'task_outputs': task_outputs,
            **losses
        }

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

UBTModel = UniversalBehavioralTransformer 