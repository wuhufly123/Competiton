from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple
import logging
from .dataset import BehaviorSequenceDataset
from ..config import Config

logger = logging.getLogger(__name__)

def create_data_loaders(data_dir: Path, config: Config, test_mode: bool = False) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    # 创建数据集
    train_dataset = BehaviorSequenceDataset(
        data_dir=data_dir,
        config=config,
        is_training=True,
        test_mode=test_mode
    )

    # 如果数据集为空，创建一个简单的测试数据集
    if len(train_dataset) == 0:
        logger.warning("训练数据集为空，创建简单测试数据集")
        raise ValueError("Empty train dataset")

    val_dataset = BehaviorSequenceDataset(
        data_dir=data_dir,
        config=config,
        is_training=False,
        test_mode=True
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader 