import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from .config import Config
from .model import UniversalBehavioralTransformer
from .data_processor import create_data_loaders
import torch.nn.functional as F
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC

logger = logging.getLogger(__name__)

class UBTTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 检查 CUDA 可用性
        self.use_cuda = torch.cuda.is_available() and self.config.device.startswith("cuda")
        
        # 设置设备
        self.device = torch.device(self.config.device)
        
        # 初始化优化器 - 使用较低的学习率以提高稳定性
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate * 0.5,  # 降低学习率
            weight_decay=config.weight_decay,
            eps=1e-8  # 提高数值稳定性
        )
        
        # 使用更稳定的学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,  # 增加耐心
            verbose=True,
            min_lr=1e-6  # 设置最小学习率
        )
            
        # 将模型移动到指定设备
        self.model = self.model.to(self.device)
        
        # 初始化评估指标
        self.metrics = {
            'churn': BinaryAUROC(),
            'category_propensity': MulticlassAUROC(num_classes=config.num_categories),
            'product_propensity': MulticlassAUROC(num_classes=config.num_products)
        }
        
        # 将指标移动到设备
        for metric in self.metrics.values():
            metric.to(self.device)
        
        # 训练状态跟踪
        self.epochs_without_improvement = {
            'churn_loss': 0,
            'category_propensity_loss': 0,
            'product_propensity_loss': 0
        }
        
        # 记录最佳损失
        self.best_losses = {
            'churn_loss': float('inf'),
            'category_propensity_loss': float('inf'),
            'product_propensity_loss': float('inf')
        }
        
        # 记录任务的样本标签比例
        self.task_label_ratios = {
            'category_propensity': [],
            'product_propensity': []
        }
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        task_losses = {
            'churn_loss': 0.0,
            'category_propensity_loss': 0.0,
            'product_propensity_loss': 0.0
        }
        total_samples = 0
        skipped_batches = 0
        
        # 任务标签统计
        labeled_counts = {
            'category_propensity': 0,
            'product_propensity': 0 
        }
        total_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            logger.info(f"Batch {batch_idx}: 开始训练")
            try:
                # 将数据移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 检查输入数据是否有NaN
                has_nan = False
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        logger.warning(f"NaN detected in input tensor {k}, skipping batch")
                        has_nan = True
                        break
                
                if has_nan:
                    skipped_batches += 1
                    continue
                
                # 记录任务标签比例
                batch_size = batch['client_id'].size(0)
                total_count += batch_size
                
                # 品类倾向性标签统计
                if 'category_propensity' in batch:
                    cat_has_label = torch.sum(torch.sum(batch['category_propensity'], dim=1) > 0).item()
                    labeled_counts['category_propensity'] += cat_has_label
                    if batch_idx % 10 == 0:  # 减少日志输出频率
                        logger.info(f"Batch {batch_idx}: 品类倾向性标签比例 = {cat_has_label/batch_size*100:.2f}%")
                
                # 产品倾向性标签统计
                if 'product_propensity' in batch:
                    prod_has_label = torch.sum(torch.sum(batch['product_propensity'], dim=1) > 0).item()
                    labeled_counts['product_propensity'] += prod_has_label
                    if batch_idx % 10 == 0:  # 减少日志输出频率
                        logger.info(f"Batch {batch_idx}: 产品倾向性标签比例 = {prod_has_label/batch_size*100:.2f}%")
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播 - 简化，不使用混合精度
                outputs = self.model(batch)
                
                loss = outputs['loss']
                logger.info(f"loss: {loss}")
                
                # 记录各任务损失
                for task in task_losses.keys():
                    if task in outputs:
                        task_losses[task] += outputs[task].item() * batch_size
                
                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN or Inf loss detected in batch {batch_idx}, skipping")
                    # 原来的行为是 continue，现在我们尝试继续，但不使用这个有问题的损失来更新参数
                    # skipped_batches += 1
                    # continue
                else:
                    # 反向传播
                    loss.backward()

                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # 检查梯度是否有无限值或NaN，并将其置为零
                    valid_gradients = True
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                logger.warning(f"Invalid gradients detected in {name}, setting to zero")
                                param.grad = torch.zeros_like(param.grad) # 将无效梯度置为零
                                # 原来的行为是跳过批次并清零所有梯度，现在只处理有问题的梯度
                                # valid_gradients = False
                                # break

                    # 原来的 valid_gradients 检查不再需要，因为我们已经处理了无效梯度
                    # if not valid_gradients:
                    #     self.optimizer.zero_grad()  # 清零梯度
                    #     skipped_batches += 1
                    #     continue
                    
                    # 更新参数
                    self.optimizer.step()
                    
                    total_loss += loss.item() * len(batch['client_id'])
                    total_samples += len(batch['client_id'])
                
            except RuntimeError as e:
                logger.error(f"RuntimeError in batch {batch_idx}: {str(e)}")
                skipped_batches += 1
                continue
        
        if skipped_batches > 0:
            logger.warning(f"Skipped {skipped_batches} batches during training")
        logger.info(f"total_loss: {total_loss}, total_samples: {total_samples}")
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        logger.info(f"Epoch average loss: {avg_loss:.4f}")
        
        # 计算并记录任务标签比例
        for task, count in labeled_counts.items():
            ratio = count / (total_count + 1e-8)
            self.task_label_ratios[task].append(ratio)
            logger.info(f"训练集 {task} 标签比例: {ratio*100:.2f}%")
        
        # 返回损失字典
        avg_task_losses = {}
        for task, loss_sum in task_losses.items():
            avg_task_losses[task] = loss_sum / total_samples if total_samples > 0 else float('inf')
            logger.info(f"Epoch average {task}: {avg_task_losses[task]:.4f}")
        
        return avg_loss, avg_task_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        task_losses = {
            'churn_loss': 0.0,
            'category_propensity_loss': 0.0,
            'product_propensity_loss': 0.0
        }
        total_samples = 0
        
        task_metrics = {
            'churn': {'preds': [], 'targets': []},
            'category_propensity': {'preds': [], 'targets': [], 'mask': []},
            'product_propensity': {'preds': [], 'targets': [], 'mask': []}
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # 将数据移到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 检查输入数据是否有NaN
                has_nan = False
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        has_nan = True
                        break
                
                if has_nan:
                    continue
                
                # 前向传播
                outputs = self.model(batch)
                
                # 检查损失是否为NaN
                if torch.isnan(outputs['loss']) or torch.isinf(outputs['loss']):
                    continue
                
                # 计算损失
                loss = outputs['loss']
                total_loss += loss.item() * batch['client_id'].size(0)
                total_samples += batch['client_id'].size(0)
                
                # 记录各任务损失
                for task in task_losses.keys():
                    if task in outputs:
                        task_losses[task] += outputs[task].item() * batch['client_id'].size(0)
                
                # 收集预测和目标
                for task, task_key in [('churn', 'churn'), 
                                    ('category_propensity', 'category_propensity'),
                                    ('product_propensity', 'product_propensity')]:
                    if task_key in outputs['task_outputs'] and task_key in batch:
                        # 确保数值稳定性
                        preds = outputs['task_outputs'][task_key]
                        if torch.isnan(preds).any() or torch.isinf(preds).any():
                            continue
                        
                        # 保存预测和目标
                        task_metrics[task]['preds'].append(preds.detach().cpu())
                        task_metrics[task]['targets'].append(batch[task_key].detach().cpu())
                        
                        # 对于品类和产品倾向性，需要记录样本是否有标签
                        if task in ['category_propensity', 'product_propensity']:
                            # 创建掩码表示哪些样本有标签（正样本或负样本）
                            # 正样本标记为1.0，负样本标记为0.0001（通过get_*_propensity_target设置）
                            has_label = (torch.sum(batch[task_key] > 0, dim=1) > 0).float()
                            task_metrics[task]['mask'].append(has_label.detach().cpu())
        
        # 计算指标
        metrics = {
            'val_loss': total_loss / total_samples if total_samples > 0 else float('inf')
        }
        
        # 添加任务损失到指标
        for task, loss_sum in task_losses.items():
            metrics[task] = loss_sum / total_samples if total_samples > 0 else float('inf')
        
        # 处理每个任务的评估指标
        for task in task_metrics:
            if task_metrics[task]['preds'] and len(task_metrics[task]['preds']) > 0:
                try:
                    preds = torch.cat(task_metrics[task]['preds'])
                    targets = torch.cat(task_metrics[task]['targets'])
                    
                    if task == 'churn':
                        # 二分类任务
                        preds = torch.sigmoid(torch.clamp(preds, min=-10, max=10))
                        metrics[f"{task}_auc"] = self.metrics[task](preds, targets).item()
                    else:
                        # 多类别任务 - 只评估有标签的样本
                        preds = torch.sigmoid(torch.clamp(preds, min=-10, max=10))
                        
                        # 获取有标签样本的掩码
                        if 'mask' in task_metrics[task] and task_metrics[task]['mask']:
                            mask = torch.cat(task_metrics[task]['mask'])
                            has_labels = mask.sum().item() > 0
                            
                            if has_labels:
                                # 只评估有标签的样本
                                # 计算有标签样本的准确率
                                has_label_indices = torch.nonzero(mask).squeeze()
                                if has_label_indices.dim() > 0:  # 确保有多个样本
                                    labeled_preds = preds[has_label_indices]
                                    labeled_targets = targets[has_label_indices]
                                    
                                    # 计算有标签样本的准确率
                                    metrics[f"{task}_accuracy"] = (
                                        (labeled_preds > 0.5).float() == labeled_targets
                                    ).float().mean().item()
                                    
                                    # 计算正样本的准确率
                                    pos_indices = torch.nonzero(labeled_targets.sum(dim=1)).squeeze()
                                    if pos_indices.dim() > 0 and pos_indices.numel() > 0:
                                        pos_preds = labeled_preds[pos_indices]
                                        pos_targets = labeled_targets[pos_indices]
                                        metrics[f"{task}_pos_accuracy"] = (
                                            (pos_preds > 0.5).float() == pos_targets
                                        ).float().mean().item()
                            else:
                                metrics[f"{task}_accuracy"] = 0.0
                                metrics[f"{task}_pos_accuracy"] = 0.0
                        else:
                            # 简化计算，避免可能的数值问题
                            metrics[f"{task}_accuracy"] = (
                                (preds > 0.5).float() == targets
                            ).float().mean().item()
                except Exception as e:
                    logger.error(f"Error computing metrics for {task}: {str(e)}")
        
        return metrics
    
    def train(self, train_loader: Optional[DataLoader] = None, val_loader: Optional[DataLoader] = None):
        """训练模型"""
        best_val_loss = float('inf')
        patience_counter = 0
        best_task_metrics = {
            'churn_auc': 0.0,
            'category_propensity_accuracy': 0.0,
            'product_propensity_accuracy': 0.0
        }
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # 训练一个epoch
            train_loss, train_task_losses = self.train_epoch(train_loader)
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics['val_loss']
                
                # 检查各任务损失是否改善
                for task, loss in val_metrics.items():
                    if task in self.best_losses:
                        if loss < self.best_losses[task]:
                            self.best_losses[task] = loss
                            self.epochs_without_improvement[task] = 0
                            logger.info(f"{task} improved: {loss:.4f}")
                        else:
                            self.epochs_without_improvement[task] += 1
                            logger.info(f"{task} no improvement for {self.epochs_without_improvement[task]} epochs")
                
                # 记录最佳任务指标
                for metric_name in best_task_metrics:
                    if metric_name in val_metrics and val_metrics[metric_name] > best_task_metrics[metric_name]:
                        best_task_metrics[metric_name] = val_metrics[metric_name]
                        logger.info(f"New best {metric_name}: {best_task_metrics[metric_name]:.4f}")
                
                # 更新学习率
                self.scheduler.step(val_loss)
                
                # 早停检查 - 基于总体损失
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_model.pt')
                    logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info("Early stopping triggered")
                        break
                
                # 检查特定任务是否没有改善
                if all(self.epochs_without_improvement[task] >= max(5, self.config.patience // 2) for task in ['category_propensity_loss', 'product_propensity_loss']):
                    logger.info("品类和产品倾向性任务长时间没有改善，提前停止训练")
                    break
                
                # 记录指标
                logger.info(f"Validation metrics: {val_metrics}")
                
                # 打印品类倾向性和产品倾向性的标签比例趋势
                for task, ratios in self.task_label_ratios.items():
                    if ratios:
                        latest_ratio = ratios[-1]
                        avg_ratio = sum(ratios) / len(ratios)
                        logger.info(f"{task} 标签比例: 当前={latest_ratio*100:.2f}%, 平均={avg_ratio*100:.2f}%")
    
    def generate_embeddings(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """生成用户表示"""
        self.model.eval()
        embeddings = []
        client_ids = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating embeddings"):
                # 将数据移到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 检查输入数据是否有NaN
                has_nan = False
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        has_nan = True
                        break
                
                if has_nan:
                    continue
                
                # 前向传播
                try:
                    outputs = self.model(batch)
                    
                    # 检查输出是否有NaN
                    if torch.isnan(outputs['user_embedding']).any():
                        continue
                    
                    # 收集用户表示和ID
                    embeddings.append(outputs['user_embedding'].cpu().numpy())
                    client_ids.append(batch['client_id'].cpu().numpy())
                except Exception as e:
                    logger.error(f"Error generating embeddings: {str(e)}")
                    continue
        
        # 合并结果
        if embeddings and client_ids:
            embeddings = np.concatenate(embeddings, axis=0)
            client_ids = np.concatenate(client_ids, axis=0)
            return client_ids, embeddings
        else:
            return np.array([]), np.array([])
