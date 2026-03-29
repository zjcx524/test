#!/bin/bash
set -ex
export TOKENIZERS_PARALLELISM=false

# 默认模型路径
MODEL_PATH="/share/models/meta-llama/Llama-3.2-1B-Instruct/"

benchmark() {
  # 默认参数
  results_folder="./dataset_result"
  model="$MODEL_PATH"
  
  # 数据集参数
  # sharegpt,mix,aps,arc,evol   
  dataset_name="aps"
  # dataset_path="/share/lq/datasets/EvolInstruct-Code-80k.json"
  # dataset_path="/share/lq/datasets/archive_train.json"
  dataset_path="/share/lq/datasets/Aps_train.json"
  # dataset_path="/share/lq/datasets/mixed_dataset_3000.json"
  # dataset_path="/share/lq/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
  
  # 其他参数
  num_prompts=100
  qps=inf
  max_concurrency=10
  prefix_len=0
  output_len=16
  tag="./results/disaggregated_pd_sglang"

  # 启动 benchmark
  python3 /home/lq/sglang/benchmark/hicache/bench_serving.py \
          --model $model \
          --backend sglang \
          --host 0.0.0.0 \
          --port 8000 \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --fixed-output-len $output_len \
          --num-prompts $num_prompts \
          --disable-shuffle \
          --request-rate "$qps" \
          --output-file "$tag"-qps-"$qps"-"$num_prompts"-"$output_len".json

  sleep 2
}

main() {
  # 启动服务器进程
  SERVER_PID=$!
  
  # 执行 benchmark
  benchmark
  
  # 停止进程
  kill $SERVER_PID
  pkill -f sglang
  pkill -f python
  pkill -f curl
}

main "$@"
