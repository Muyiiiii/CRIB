#!/bin/bash

# define datasets
# datasets=("PEMS" "Elec" "ETTh1" "Metr" "PEMS_imputed" "Elec_imputed" "ETTh1_imputed" "Metr_imputed")
datasets=("ETTh1")

# define missing rates
# missing_rates=(0.2 0.4 0.6 0.7)
missing_rates=(0.2)

# define loss types
# iters=("1" "2" "3")
iters=("1")

models=("CRIB" "DLinear" "SegRNN" "Transformer" "iTransformer" "PatchTST" "TSMixer" "WPMixer" "PAttn")
# models=("CRIB" "PatchTST" )

# missing_patterns=("point" "block" "col")
missing_patterns=("point")

# define training epochs and learning rate
train_epochs=10
learning_rate=0.001
# remove the fixed batch_size definition here, it will be dynamically set in the loop
batch_size=32 
model_dim=32
seed=123

date="250921"

# csv_path="./v${version}_${date}_missing_pattern_result.csv"
csv_path="./result/${date}_result.csv"

# embedding size
# outer loop to iterate over datasets
for dataset in "${datasets[@]}"; do
    # inner loop to iterate over missing patterns
    for missing_pattern in "${missing_patterns[@]}"; do
        for model in "${models[@]}"; do
            # inner loop to iterate over missing rates
            for missing_rate in "${missing_rates[@]}"; do
                for iter in "${iters[@]}"; do
                    # create a log file for each dataset
                    log_file="./log/${date}/${dataset}-${missing_pattern}-${missing_rate}-${model}-${iter}_training_24steps.log"
                    # clear old log file or create a new one
                    mkdir -p "$(dirname "$log_file")"
                    
                    > "$log_file"
                    echo "Starting training on $dataset with missing pattern $missing_pattern, missing rate $missing_rate, model $model, batch_size $batch_size, and iter $iter" | tee -a $log_file
                    # python train.py --dataset $dataset --model $model --batch_size $batch_size --missing_pattern $missing_pattern --missing_rate $missing_rate --seq_len 24 --pred_len 24 --train_epochs $train_epochs --iter $iter --model $model --csv_path $csv_path --seed $seed --exp_type 'Model' 2>&1 | tee -a $log_file
                    python train.py \
                        --dataset $dataset \
                        --model $model \
                        --batch_size $batch_size \
                        --missing_pattern $missing_pattern \
                        --missing_rate $missing_rate \
                        --seq_len 24 \
                        --pred_len 24 \
                        --model_dim $model_dim \
                        --train_epochs $train_epochs \
                        --iter $iter \
                        --csv_path $csv_path \
                        --exp_type 'Train' \
                        --seed $seed \
                        2>&1 | tee -a $log_file
                    echo "Finished training on $dataset with missing pattern $missing_pattern, missing rate $missing_rate, model $model, and iter $iter" | tee -a $log_file
                done
            done
        done
    done
done

echo "All training processes are completed."