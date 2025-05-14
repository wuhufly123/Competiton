LOG_NAME="exp_128_test"
OUTPUT_DIR="./logs"  # 默认输出目录
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/$LOG_NAME$(date +%Y%m%d_%H%M%S).log"

python -m ubt_solution.create_embeddings \
    --data-dir /data/mhwang/Rec/RecSys/recsys2025/data \
    --embeddings-dir /data/mhwang/Rec/RecSys/recsys2025/submit_file/exp_128_test \
    --accelerator cuda \
    --devices 2 \
    --num-workers 4 \
    --batch-size 2048 \
    --num-epochs 20 \
    --learning-rate 1e-4 \
    --test-mode \
    --task-weights "churn:1.0,category_propensity:0.0,product_propensity:0.0" \
    > >(tee -a "$LOG_FILE") 2>&1
# 检查命令执行是否成功
if [ $? -eq 0 ]; then
    echo "训练完成，日志已保存至: $LOG_FILE"
else
    echo "训练失败，请检查日志: $LOG_FILE"
fi