#!/bin/bash

# 默认参数
DATA_DIR="/data/mhwang/Rec/RecSys/recsys2025/data"
EMBEDDINGS_DIR="/data/mhwang/Rec/RecSys/recsys2025/submit_file/concat_file"
TASKS="churn propensity_category propensity_sku"
LOG_NAME="exp_concat"
ACCELERATOR="gpu"
DEVICES="0"
OUTPUT_DIR="./logs"  # 默认输出目录


# nohup python -m training_pipeline.train --data-dir /data/mhwang/Rec/RecSys/recsys2025/data --embeddings-dir /data/mhwang/Rec/RecSys/recsys2025/submit_file/ubt_embed --tasks churn propensity_category propensity_sku --log-name ubt_exp --accelerator gpu --devices 0 --disable-relevant-clients-check
# 解析命令行参数

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# 设置日志文件路径
LOG_FILE="$OUTPUT_DIR/$LOG_NAME$(date +%Y%m%d_%H%M%S).log"

# 执行命令并将输出同时写入日志文件和终端
python -m training_pipeline.train \
    --data-dir "$DATA_DIR" \
    --embeddings-dir "$EMBEDDINGS_DIR" \
    --tasks $TASKS \
    --log-name "$LOG_NAME" \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    --disable-relevant-clients-check \
    > >(tee -a "$LOG_FILE") 2>&1

# 检查命令执行是否成功
if [ $? -eq 0 ]; then
    echo "训练完成，日志已保存至: $LOG_FILE"
else
    echo "训练失败，请检查日志: $LOG_FILE"
fi