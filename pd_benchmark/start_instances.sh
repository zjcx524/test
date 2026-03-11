#!/bin/bash
# One-click launcher for SGLang prefill/decode disaggregation + router
# 
set -e
# "Qwen/Qwen3-0.6B"
MODEL="/share/models/meta-llama/Llama-3.2-1B-Instruct/"
#/share/models/meta-llama/Llama-3.2-1B-Instruct/

start_prefill() {
  echo "========== Starting Prefill Servers =========="
  
  echo "Starting SGLang Prefill Server 1 (port 30000) on GPU 0,1..."
  python -m sglang.launch_server \
    --model-path $MODEL \
    --disaggregation-mode prefill \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --disaggregation-decode-tp 1 \
    --base-gpu-id 0 \
    --chunked-prefill-size -1 \
    --port 10000 \
    --disable-cuda-graph \
    --mem-fraction-static 0.9 \
    --max-total-tokens 10000 \
    --disaggregation-transfer-backend nixl &

  # echo "Starting SGLang Prefill Server 2 (port 30001) on GPU 4,5..."
  # python -m sglang.launch_server \
  #   --model-path $MODEL \
  #   --disaggregation-mode prefill \
  #   --tensor-parallel-size 2 \
  #   --pipeline-parallel-size 1 \
  #   --disaggregation-decode-tp 1 \
  #   --base-gpu-id 4 \
  #   --chunked-prefill-size -1 \
  #   --port 20001 \
  #   --disable-cuda-graph \
  #   --mem-fraction-static 0.9 \
  #   --max-total-tokens 10000 \
  #   --disaggregation-transfer-backend nixl &

  echo "Prefill servers started."
}

start_decode() {
  echo "========== Starting Decode Servers =========="
  
  echo "Starting SGLang Decode Server 1 (port 30002) on GPU 2..."
  python -m sglang.launch_server \
    --model-path $MODEL \
    --disaggregation-mode decode \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --disaggregation-prefill-pp 1 \
    --base-gpu-id 2 \
    --chunked-prefill-size -1 \
    --port 30002 \
    --disable-cuda-graph \
    --mem-fraction-static 0.9 \
    --max-total-tokens 10000 \
    --disaggregation-transfer-backend nixl &

  # echo "Starting SGLang Decode Server 2 (port 30003) on GPU 3..."
  # python -m sglang.launch_server \
  #   --model-path $MODEL \
  #   --disaggregation-mode decode \
  #   --tensor-parallel-size 1 \
  #   --pipeline-parallel-size 1 \
  #   --disaggregation-prefill-pp 1 \
  #   --base-gpu-id 3 \
  #   --chunked-prefill-size -1 \
  #   --port 40003 \
  #   --disable-cuda-graph \
  #   --mem-fraction-static 0.9 \
  #   --max-total-tokens 10000 \
  #   --disaggregation-transfer-backend nixl &

  # echo "Starting SGLang Decode Server 3 (port 30004) on GPU 6..."
  # python -m sglang.launch_server \
  #   --model-path $MODEL \
  #   --disaggregation-mode decode \
  #   --tensor-parallel-size 1 \
  #   --pipeline-parallel-size 1 \
  #   --disaggregation-prefill-pp 1 \
  #   --base-gpu-id 6 \
  #   --chunked-prefill-size -1 \
  #   --port 50004 \
  #   --disable-cuda-graph \
  #   --mem-fraction-static 0.9 \
  #   --max-total-tokens 10000 \
  #   --disaggregation-transfer-backend nixl &

  # echo "Starting SGLang Decode Server 4 (port 30005) on GPU 7..."
  # python -m sglang.launch_server \
  #   --model-path $MODEL \
  #   --disaggregation-mode decode \
  #   --tensor-parallel-size 1 \
  #   --pipeline-parallel-size 1 \
  #   --disaggregation-prefill-pp 1 \
  #   --base-gpu-id 7 \
  #   --chunked-prefill-size -1 \
  #   --port 60005 \
  #   --disable-cuda-graph \
  #   --mem-fraction-static 0.9 \
  #   --max-total-tokens 10000 \
  #   --disaggregation-transfer-backend nixl &

  echo "Decode servers started."
}

start_router() {
  echo "========== Starting Router =========="
  
  echo "Starting Router (port 8000)..."
  python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:10000 \
    --decode http://127.0.0.1:30002 \
    --host 0.0.0.0 \
    --port 8000 &
    # --prefill-policy round_robin \
    # --decode-policy round_robin \、
    # --prefill http://127.0.0.1:10000 \
    # --prefill http://127.0.0.1:20001 \
    # --decode http://127.0.0.1:30002 \
    # --decode http://127.0.0.1:40003 \
    # --decode http://127.0.0.1:50004 \
    # --decode http://127.0.0.1:60005 \

  echo "Router started."
}

# 按顺序启动所有服务
start_prefill

echo "Waiting for prefill servers to initialize..."
sleep 50

start_decode

echo "Waiting for decode servers to initialize..."
sleep 100

start_router

echo "=========================================="
echo "All services started in background."
echo "Use 'ps aux | grep sglang' to check status."
echo "Use 'pkill -f sglang' to stop all services."
echo "=========================================="
