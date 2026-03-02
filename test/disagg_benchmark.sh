#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "?? Warning: The usage of disaggregated prefill is experimental and subject to change ??"
sleep 1

# meta-llama/Meta-Llama-3.1-8B-Instruct or deepseek-ai/DeepSeek-V2-Lite
MODEL_NAME="/home/ljh1/models/meta-llama/Llama-3.2-1B-Instruct"

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

benchmark() {
  results_folder="./dataset_result"
  model="$MODEL_NAME"
  #  arc aps evol mix sharegpt custom(apsΪ���ı�Ӧ���ſ�goodput)
  # dataset_name="custom"
  dataset_name="evol"
  # dataset_path="./datasets/archive_train.json"
  # dataset_path="./datasets/Aps_train.json"
  dataset_path="./datasets/EvolInstruct-Code-80k.json"
  # dataset_path="./datasets/mixed_dataset_3000.json"
  # dataset_path="./datasets/ShareGPT_V3_unfiltered_cleaned_split.json"

  # dataset_path="./dataset/part_Evol50.json"
  # 300 200 100 16
  num_prompts=10
  qps=10
  prefix_len=0
  output_len=16
  tag=vllm-$dataset_name-$num_prompts-$output_len

vllm bench serve \
    --backend vllm \
    --model $model \
    --dataset-name $dataset_name \
    --dataset-path $dataset_path \
    --sharegpt-output-len $output_len \
    --num-prompts $num_prompts \
    --port 10001 \
    --save-result \
    --result-dir $results_folder \
    --result-filename "$tag"-qps-"$qps".json \
    --request-rate "$qps" \
    --max-concurrency 8 \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --skip-chat-template &
    # --disable-shuffle

#   python3 ../benchmark_serving_structured_output.py \
#           --backend vllm \
#           --model $model \
#           --dataset-name $dataset_name \
#           --dataset-path $dataset_path \
#           --sharegpt-output-len $output_len \
#           --num-prompts $num_prompts \
#           --port 8000 \
#           --save-result \
#           --result-dir $results_folder \
#           --result-filename "$tag"-qps-"$qps".json \
#           --request-rate "$qps"

#   python3 ../benchmark_serving_structured_output.py \
#           --backend vllm \
#           --model $model \
#           --dataset json \
#           --json-scheme-path $dataset_path \
#           --structured-output-ratio 0 \
#           --num-prompts $num_prompts \
#           --request-rate $qps \
#           --output-len $output_len \
#           --result-dir $results_folder \
#           --result-filename "$tag"-qps-"$qps".json \

  sleep 2
}

main() {
  # launch_chunked_prefill
  # for qps in 2 4 10; do
  # benchmark $qps $default_output_len chunked_prefill
  # done
  # kill_gpu_processes

  #launch_disagg_prefill
  #for qps in 2; do
  #benchmark $qps $default_output_len disagg_prefill
  #done
  # qps=2
  # for output_len in 16 32 64 128 256 512 768 1024 1100 1250 1500; do
  #   benchmark $qps $output_len tp=1pp=1-
  # done
  benchmark
  # kill_gpu_processes

  #python3 visualize_benchmark_results.py

}


main "$@"

# python benchmarks/benchmark_serving_structured_output.py \
#         --backend <backend> \
#         --model <your_model> \
#         --dataset json \
#         --structured-output-ratio 1.0 \
#         --request-rate 10 \
#         --num-prompts 1000