import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import psutil
import gc
import time
from pathlib import Path
from typing import Dict, List
from .target_data import TargetData
from .memory_utils import report_memory_usage
import pandas as pd
from .utils import (
    vectorize_text,
    create_user_chunks,
    load_events_parallel,
    load_product_properties
)
from ..config import Config

logger = logging.getLogger(__name__)

class BehaviorSequenceDataset(Dataset):
    def __init__(self, 
                 data_dir: Path,
                 config: Config,
                 is_training: bool = True,
                 test_mode: bool = False):
        self.config = config
        self.is_training = is_training
        self.test_mode = test_mode
        
        report_memory_usage("初始化数据集开始")
        
        # 加载相关用户ID
        relevant_clients_path = data_dir / "input" / "relevant_clients.npy"
        logger.info(f"正在加载相关用户ID，文件路径: {relevant_clients_path}")
        if not relevant_clients_path.exists():
            logger.error(f"文件不存在: {relevant_clients_path}")
            raise FileNotFoundError(f"找不到文件: {relevant_clients_path}")
        self.relevant_clients = np.load(relevant_clients_path)
        
        # 如果是测试模式，只取前10000个用户
        if self.test_mode:
            self.relevant_clients = self.relevant_clients[:10000]
            logger.info(f"测试模式：只处理前 {len(self.relevant_clients)} 个用户")
        else:
            logger.info(f"加载了 {len(self.relevant_clients)} 个相关用户")
        
        # 设置用户分块大小和内存调优参数
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"系统可用内存: {available_memory_gb:.2f} GB")
        
        # 根据可用内存调整分块大小
        if available_memory_gb > 32:  # 如果有大量内存可用
            self.chunk_size = 20000 if self.test_mode else 100000
        elif available_memory_gb > 16:
            self.chunk_size = 10000 if self.test_mode else 50000
        else:  # 内存有限
            self.chunk_size = 5000 if self.test_mode else 25000
            
        logger.info(f"根据系统内存自动设置块大小为: {self.chunk_size}")
        
        self.user_chunks = create_user_chunks(self.relevant_clients, self.chunk_size)
        logger.info(f"将用户分成 {len(self.user_chunks)} 个块进行处理，每块大约 {self.chunk_size} 个用户")
        
        # 加载数据
        self.events = {}
        self.user_sequences = {}
        
        report_memory_usage("开始分块加载事件数据")
        
        # 使用分块策略加载和处理事件
        for chunk_idx, user_chunk in enumerate(self.user_chunks):
            start_time = time.time()
            logger.info(f"处理用户块 {chunk_idx+1}/{len(self.user_chunks)}")
            # 将当前块设置为相关用户集
            self.current_chunk_users = set(user_chunk)
            
            # 加载当前块用户的事件数据并处理序列
            logger.info("开始并行加载当前块的事件数据...")
            chunk_events = load_events_parallel(data_dir, self.current_chunk_users)
            
            report_memory_usage(f"块 {chunk_idx+1} 事件数据加载完成")
            
            # 加载商品属性数据（所有块共享）
            if chunk_idx == 0:
                logger.info("开始加载商品属性数据...")
                self.product_properties = load_product_properties(data_dir)
                logger.info("商品属性数据加载完成")
            
            # 构建当前块用户的序列
            logger.info(f"开始构建用户块 {chunk_idx+1} 的序列...")
            chunk_sequences = self._build_user_sequences(chunk_events, user_chunk)
            
            # 合并到总序列字典中
            self.user_sequences.update(chunk_sequences)
            
            # 计算并报告处理时间
            elapsed_time = time.time() - start_time
            logger.info(f"块 {chunk_idx+1} 处理完成，耗时: {elapsed_time:.2f} 秒")
            
            # 释放块事件数据内存并主动触发垃圾回收
            del chunk_events
            del chunk_sequences
            gc.collect()
            
            report_memory_usage(f"块 {chunk_idx+1} 处理完成")
            
        logger.info(f"用户序列构建完成，共 {len(self.user_sequences)} 个用户")
        
        # 加载目标数据
        self.target_data = TargetData(data_dir)
        
        # 用户ID列表和序列长度
        self.client_ids = list(self.user_sequences.keys())
        self.max_seq_length = self.config.max_seq_length
        
        # 创建事件类型映射
        self.event_type_map = {
            'product_buy': 0,
            'add_to_cart': 1,
            'remove_from_cart': 2,
            'page_visit': 3,
            'search_query': 4
        }
        
        # 创建商品属性查找字典，加速访问
        logger.info("创建商品属性查找字典...")
        self.product_dict = {}
        try:
            self.product_dict = self.product_properties.set_index('sku').to_dict('index')
            logger.info(f"商品属性字典创建完成，包含 {len(self.product_dict)} 个商品")
        except Exception as e:
            logger.error(f"创建商品属性字典时出错: {str(e)}")
        
        # 初始化样本缓存
        self.item_cache = {}
        # 根据可用内存调整缓存大小
        self.cache_size = min(int(available_memory_gb * 200), 5000)  # 每GB内存缓存约200个样本，但最多5000个
        logger.info(f"设置样本缓存大小为 {self.cache_size}")
        
        report_memory_usage("数据集初始化完成")
        
    def _build_user_sequences(self, events: Dict[str, List], user_chunk: List[int]) -> Dict[int, List[Dict]]:
        """构建用户序列"""
        logger.info("开始构建用户序列...")
        report_memory_usage("开始构建用户序列")
        user_chunk_set = set(user_chunk)
        
        # 检查缓存文件是否存在
        cache_dir = Path(__file__).parent / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"user_sequences_{hash(tuple(sorted(user_chunk_set)))}.pkl"
        
        if cache_file.exists():
            logger.info(f"找到缓存文件 {cache_file}，尝试加载...")
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    user_sequences = pickle.load(f)
                logger.info(f"成功从缓存加载 {len(user_sequences)} 个用户序列")
                return user_sequences
            except Exception as e:
                logger.warning(f"加载缓存文件失败: {str(e)}，将重新构建序列")
        
        # 合并所有事件到一个DataFrame
        logger.info("合并事件数据...")
        try:
            # 预先估计大小，避免多次内存分配
            total_rows = sum(len(df) for df in events.values())
            logger.info(f"预计合并 {total_rows} 行事件数据")
            
            # 只合并有数据的DataFrame
            non_empty_events = [df for df in events.values() if not df.empty]
            if not non_empty_events:
                logger.warning("没有事件数据，将返回空序列")
                return {client_id: self._create_default_sequence() for client_id in user_chunk_set}
            
            all_events = pd.concat(non_empty_events, ignore_index=True)
        except Exception as e:
            logger.error(f"合并事件数据时发生错误: {str(e)}")
            # 创建一个空DataFrame
            all_events = pd.DataFrame(columns=['client_id', 'timestamp', 'event_type', 'sku'])
        
        if len(all_events) == 0:
            logger.warning("没有事件数据，将返回空序列")
            return {client_id: self._create_default_sequence() for client_id in user_chunk_set}
        
        # 优化内存使用
        logger.info("优化数据类型以减少内存占用...")
        # 将整数列转换为最小可能的类型
        if 'client_id' in all_events.columns:
            all_events['client_id'] = pd.to_numeric(all_events['client_id'], downcast='integer')
        if 'sku' in all_events.columns:
            all_events['sku'] = pd.to_numeric(all_events['sku'], downcast='integer')
        
        # 转换时间戳
        logger.info("处理时间戳...")
        all_events['timestamp'] = pd.to_datetime(all_events['timestamp'], errors='coerce')
        all_events = all_events.dropna(subset=['timestamp'])
        
        report_memory_usage("事件数据合并和类型优化完成")
        
        # 使用分组优化处理逻辑
        logger.info("按用户分组预处理事件...")
        
        # 先找出有事件的用户
        users_with_events = all_events['client_id'].unique()
        logger.info(f"在事件数据中发现 {len(users_with_events)} 个用户")
        
        # 为没有事件的用户创建默认序列
        user_sequences = {client_id: self._create_default_sequence() 
                         for client_id in user_chunk_set if client_id not in users_with_events}
        
        # 分批处理以减少内存消耗
        max_sequence_length = self.config.max_seq_length
        
        # 根据事件数据大小动态调整批次大小
        events_per_user = len(all_events) / max(1, len(users_with_events))
        batch_size = 5000  # 增大批次大小
            
        logger.info(f"每用户平均事件数: {events_per_user:.1f}, 设置批次大小: {batch_size}")
        
        # 计算总批次数
        num_batches = (len(users_with_events) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(users_with_events))
            batch_users = users_with_events[batch_start:batch_end]
            
            logger.info(f"处理用户批次 {batch_idx+1}/{num_batches}，包含 {len(batch_users)} 个用户")
            
            # 只选择当前批次用户的事件
            batch_events = all_events[all_events['client_id'].isin(batch_users)].copy()
            
            # 按用户分组
            user_grouped_events = batch_events.groupby('client_id')
            
            # 高效处理每个用户的序列
            batch_sequences = {}
            for client_id, user_df in user_grouped_events:
                try:
                    # 按时间排序
                    user_events = user_df.sort_values('timestamp')
                    
                    # 如果序列太长，只保留最近的记录
                    if len(user_events) > max_sequence_length:
                        user_events = user_events.iloc[-max_sequence_length:]
                    
                    # 标准化时间戳
                    min_ts = user_events['timestamp'].min()
                    # 使用向量化操作，不是循环
                    user_events['norm_timestamp'] = (user_events['timestamp'] - min_ts).dt.total_seconds() / 86400.0
                    
                    # 使用列表推导式构建序列 - 比循环更快
                    sequence = [
                        {
                            'event_type': row['event_type'],
                            'timestamp': row['norm_timestamp'],
                            'sku': row['sku'],
                            **({'query': row['query']} if row['event_type'] == 'search_query' and 'query' in row else {})
                        }
                        for _, row in user_events.iterrows()
                    ]
                    
                    batch_sequences[client_id] = sequence
                except Exception as e:
                    logger.error(f"处理用户 {client_id} 序列时发生错误: {str(e)}")
                    # 出错时创建默认序列
                    batch_sequences[client_id] = self._create_default_sequence()
            
            # 更新总序列字典
            user_sequences.update(batch_sequences)
            
            # 清理批次数据以释放内存
            del batch_events, user_grouped_events, batch_sequences
            gc.collect()
            
            # 每4个批次报告一次内存使用情况
            if batch_idx % 4 == 0:
                report_memory_usage(f"用户批次 {batch_idx+1}/{num_batches} 处理完成")
        
        # 保存到缓存文件
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(user_sequences, f)
            logger.info(f"成功将 {len(user_sequences)} 个用户序列保存到缓存文件 {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存文件失败: {str(e)}")
        
        report_memory_usage("用户序列构建完成")
        logger.info(f"成功构建了 {len(user_sequences)} 个用户序列")
        return user_sequences
    
    def _create_default_sequence(self) -> List[Dict]:
        """为空序列创建默认值"""
        return [{'event_type': 'page_visit', 'timestamp': 0.0}]
    
    def __len__(self) -> int:
        return len(self.client_ids)
    
    def __getitem__(self, idx):
        # 首先检查缓存
        if idx in self.item_cache:
            return self.item_cache[idx]
        
        client_id = self.client_ids[idx]
        sequence = self.user_sequences[client_id]
        
        # 预分配内存
        event_types = torch.zeros(self.max_seq_length, dtype=torch.long)
        timestamps = torch.zeros(self.max_seq_length, dtype=torch.float)
        categories = torch.zeros(self.max_seq_length, dtype=torch.long)
        prices = torch.zeros(self.max_seq_length, dtype=torch.long)
        names = torch.zeros((self.max_seq_length, 16), dtype=torch.float)
        queries = torch.zeros((self.max_seq_length, 16), dtype=torch.float)
        mask = torch.zeros(self.max_seq_length, dtype=torch.bool)
        behavior_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
        
        # 填充序列 - 使用向量化操作替代循环
        seq_len = min(len(sequence), self.max_seq_length)
        
        # 批量处理事件类型
        for i in range(seq_len):
            event = sequence[i]
            event_type = event['event_type']
            event_code = self.event_type_map.get(event_type, 0)
            
            event_types[i] = event_code
            behavior_ids[i] = event_code
            timestamps[i] = float(event['timestamp'])
            mask[i] = True
            
            # 处理商品特征
            sku = event.get('sku', -1)
            if event_type in [self.event_type_map['product_buy'], 
                              self.event_type_map['add_to_cart'], 
                              self.event_type_map['remove_from_cart']] and sku != -1:
                if sku in self.product_dict:
                    product = self.product_dict[sku]
                    categories[i] = int(product.get('category', 0))
                    prices[i] = min(100, int(product.get('price', 0)))
                    names[i] = torch.tensor(vectorize_text(str(product.get('name', ''))))
            
            # 处理搜索查询
            if event_type == 'search_query' and 'query' in event:
                queries[i] = torch.tensor(vectorize_text(str(event['query'])))
        
        # 获取目标值 - 使用异常处理确保稳定性
        try:
            category_propensity = self.target_data.get_category_propensity_target(client_id)
            category_tensor = torch.tensor(category_propensity, dtype=torch.float)
        except Exception as e:
            logger.warning(f"获取用户 {client_id} 的品类倾向目标时出错: {str(e)}")
            category_tensor = torch.zeros(len(self.target_data.propensity_category), dtype=torch.float)
        
        try:
            product_propensity = self.target_data.get_product_propensity_target(client_id)
            product_tensor = torch.tensor(product_propensity, dtype=torch.float)
        except Exception as e:
            logger.warning(f"获取用户 {client_id} 的商品倾向目标时出错: {str(e)}")
            product_tensor = torch.zeros(len(self.target_data.propensity_sku), dtype=torch.float)
        
        try:
            churn_target = self.target_data.get_churn_target(client_id)
            churn_tensor = torch.tensor(churn_target, dtype=torch.float)
        except Exception as e:
            logger.warning(f"获取用户 {client_id} 的流失目标时出错: {str(e)}")
            churn_tensor = torch.tensor(0.0, dtype=torch.float)
        
        # 检查并替换NaN
        if torch.isnan(category_tensor).any():
            category_tensor = torch.zeros_like(category_tensor)
        if torch.isnan(product_tensor).any():
            product_tensor = torch.zeros_like(product_tensor)
            
        # 构建结果
        result = {
            'event_types': event_types,
            'timestamps': timestamps,
            'categories': categories,
            'prices': prices,
            'names': names,
            'queries': queries,
            'mask': mask,
            'behavior_ids': behavior_ids,
            'client_id': torch.tensor(client_id),
            'churn': churn_tensor,
            'category_propensity': category_tensor,
            'product_propensity': product_tensor
        }
        
        # 清理任何NaN值
        for key, value in result.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                result[key] = torch.where(torch.isnan(value), torch.zeros_like(value), value)
        
        # 缓存结果，如果缓存过大则移除最早项
        if len(self.item_cache) >= self.cache_size:
            # 简单策略：随机移除一个项
            remove_key = list(self.item_cache.keys())[0]
            del self.item_cache[remove_key]
        
        self.item_cache[idx] = result
        return result 