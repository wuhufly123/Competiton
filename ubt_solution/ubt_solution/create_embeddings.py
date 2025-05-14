import argparse
import logging
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from .config import Config
from .trainer import UBTTrainer
from .data_processor import create_data_loaders
from .model import UniversalBehavioralTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory where target and input data are stored",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory where to store generated embeddings",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        help="Accelerator type (cuda or cpu)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Device ID",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Whether to use test mode (process only the first 100 users)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--task-weights",
        type=str,
        default=None,
        help="Task weights in format 'churn:1.0,category_propensity:0.5,product_propensity:0.5'",
    )
    return parser

def save_embeddings(embeddings_dir: Path, client_ids: np.ndarray, embeddings: np.ndarray):
    """保存嵌入向量和客户端ID"""
    logger.info("Saving embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    np.save(embeddings_dir / "client_ids.npy", client_ids)

def main():
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 创建目录
    data_dir = Path(args.data_dir)
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析任务权重
    task_weights = None
    if args.task_weights:
        task_weights = {}
        for pair in args.task_weights.split(','):
            key, value = pair.split(':')
            task_weights[key] = float(value)
        logger.info(f"Using custom task weights: {task_weights}")
    
    # 加载配置
    config = Config(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        accelerator="cuda",
        devices=[int(args.devices)],
        num_workers=args.num_workers,
        output_dir=str(embeddings_dir),
        device=f"cuda:{args.devices}",
        task_weights=task_weights
    )
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(data_dir, config, args.test_mode)
    
    # 初始化模型
    model = UniversalBehavioralTransformer(config)
    
    # 初始化训练器
    trainer = UBTTrainer(
        model=model,
        config=config
    )
    
    # 训练模型
    logger.info("开始训练模型...")
    trainer.train(train_loader=train_loader, val_loader=val_loader)
    
    # 生成用户嵌入向量
    logger.info("生成用户嵌入向量...")
    client_ids, embeddings = trainer.generate_embeddings(train_loader)
    
    # 保存嵌入向量
    save_embeddings(embeddings_dir, client_ids, embeddings)
    
    logger.info("完成！")

if __name__ == "__main__":
    main()
