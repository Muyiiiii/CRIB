#!/bin/bash
# rebuttal 原有baseline上的数据集扩展实验

# === 1. 实验配置 ===

# 定义数据集
# datasets=("PEMS" "Elec" "ETTh1" "Metr" "PEMS_imputed" "Elec_imputed" "ETTh1_imputed" "Metr_imputed")
# datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "Elec" "PEMS" "Metr" "BeijingAir" "Exchange" "Illness" "Traffic" "Weather")
datasets=("ETTm1" "ETTm2" "Elec" "PEMS" "Metr" "BeijingAir" "Exchange" "Illness" "Traffic" "Weather")

# 定义缺失率
missing_rates=(0.0 0.2 0.4 0.6 0.7)
# missing_rates=(0.2)

# 定义迭代次数 (例如，用于多次运行取平均值)
# iters=("1" "2" "3")
iters=("1")

# 定义模型 — 注释掉不需要跑的，留下需要的
# TSL forecasters: CRIB DLinear SegRNN Transformer iTransformer PatchTST TSMixer WPMixer PAttn TimesNet TimeXer KANAD MultiPatchFormer FreTS
# PyPOTS imputation models: CSDI ImputeFormer
# Other: NeuralCDE
models=("CRIB" "DLinear" "SegRNN" "iTransformer" "PatchTST" "TSMixer" "WPMixer" "PAttn" "TimesNet" "TimeXer" "NeuralCDE" "CSDI" "ImputeFormer")
# models=("CRIB")

# 定义缺失模式
missing_patterns=("point" "block" "col")
# missing_patterns=("point")

# === 2. 训练参数 ===
train_epochs=10
learning_rate=0.001
batch_size=16 
model_dim=32
seq_len=24
seed=0

# === 3. 路径和日期设置 ===
date="2025_11_16"
csv_path="./result/1_rebuttal_more_datasets/${date}_rebuttal_more_datasets_result.csv"
log_dir="./log/1_rebuttal_more_datasets/${date}"

# 确保日志和结果目录存在
mkdir -p "$(dirname "$csv_path")"
mkdir -p "$log_dir"

# === 4. 实验循环 ===
# 循环顺序: missing_pattern -> dataset -> missing_rate -> model -> iter

echo "--- 实验开始 ---"
echo "日志目录: $log_dir"
echo "结果CSV: $csv_path"

for missing_pattern in "${missing_patterns[@]}"; do
    for dataset in "${datasets[@]}"; do
        for missing_rate in "${missing_rates[@]}"; do
            for model in "${models[@]}"; do
                for iter in "${iters[@]}"; do
                    
                    # 定义日志文件名
                    log_file="${log_dir}/${dataset}-${missing_pattern}-${missing_rate}-${model}-${iter}_training_${seq_len}steps.log"
                    
                    # 清空/创建日志文件
                    > "$log_file"
                    
                    # 打印开始信息到控制台和日志
                    echo "---" | tee -a "$log_file"
                    echo "开始: Pattern=$missing_pattern, Dataset=$dataset, Rate=$missing_rate, Model=$model, Iter=$iter" | tee -a "$log_file"
                    echo "日志文件: $log_file" | tee -a "$log_file"
                    
                    # 执行 Python 训练脚本
                    # 2>&1 将标准错误 (stderr) 重定向到标准输出 (stdout)
                    # | tee -a "$log_file" 将 stdout 附加到日志文件，并同时在终端显示
                    python train.py \
                        --dataset "$dataset" \
                        --model "$model" \
                        --batch_size "$batch_size" \
                        --missing_pattern "$missing_pattern" \
                        --missing_rate "$missing_rate" \
                        --seq_len "$seq_len" \
                        --pred_len 24 \
                        --learning_rate "$learning_rate" \
                        --model_dim "$model_dim" \
                        --train_epochs "$train_epochs" \
                        --iter "$iter" \
                        --csv_path "$csv_path" \
                        --exp_type 'Train' \
                        --seed "$seed" \
                        | tee -a "$log_file"
                        # 2>&1 | tee -a "$log_file"
                    
                    echo "完成: Pattern=$missing_pattern, Dataset=$dataset, Rate=$missing_rate, Model=$model, Iter=$iter" | tee -a "$log_file"

                done
            done
        done
    done
done

echo "---"
echo "所有训练过程已全部完成。"