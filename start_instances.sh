#!/bin/bash
# One-click launcher for SGLang prefill/decode disaggregation + router
# 
set -e

MODEL="Qwen/Qwen3-0.6B"


# prefill t2p1 decode t1p2

echo "Starting SGLang Prefill Server (port 30000)..."
python -m sglang.launch_server \
  --model-path $MODEL \
  --disaggregation-mode prefill \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --disaggregation-decode-tp 1 \
  --base-gpu-id 0 \
  --chunked-prefill-size -1 \
  --port 30000 \
  --disable-cuda-graph \
  --attention-backend triton \
  --mem-fraction-static 0.7 \
  --max-total-tokens 10000 &

echo "Starting SGLang Decode Server (port 30001) on GPU1..."
python -m sglang.launch_server \
  --model-path $MODEL \
  --disaggregation-mode decode \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --disaggregation-prefill-pp 1 \
  --base-gpu-id 1 \
  --chunked-prefill-size -1 \
  --port 30002 \
  --attention-backend triton \
  --disable-cuda-graph \
  --mem-fraction-static 0.7 \
  --max-total-tokens 10000  &


# prefill t2 decode t1

# echo "Starting SGLang Prefill Server (port 30000)..."
# python -m sglang.launch_server \
#   --model-path $MODEL \
#   --disaggregation-mode prefill \
#   --tensor-parallel-size 2 \
#   --disaggregation-decode-tp 1 \
#   --base-gpu-id 0 \
#   --chunked-prefill-size -1 \
#   --port 30000 \
#   --disable-cuda-graph \
#   --mem-fraction-static 0.7 \
#   --max-total-tokens 10000 \
#   --disaggregation-transfer-backend nixl &

# echo "Starting SGLang Decode Server (port 30001) on GPU1..."
# python -m sglang.launch_server \
#   --model-path $MODEL \
#   --disaggregation-mode decode \
#   --tensor-parallel-size 1 \
#   --base-gpu-id 2 \
#   --chunked-prefill-size -1 \
#   --port 30002 \
#   --disable-cuda-graph \
#   --mem-fraction-static 0.7 \
#   --max-total-tokens 10000 \
#   --disaggregation-transfer-backend nixl  &



  # Prefill
  # --disaggregation-decode-tp 1 \
  # Decode
  # --disaggregation-prefill-pp 2 \

# echo "Starting SGLang Prefill Server (port 30000)..."
# python -m sglang.launch_server \
#   --model-path $MODEL \
#   --disaggregation-mode prefill \
#   --tensor-parallel-size 2 \
#   --disaggregation-decode-tp 1 \
#   --data-parallel-size 1 \
#   --disaggregation-decode-dp 2 \
#   --base-gpu-id 0 \
#   --chunked-prefill-size -1 \
#   --port 30000 \
#   --disable-cuda-graph \
#   --mem-fraction-static 0.7 \
#   --max-total-tokens 10000 \
#   --load-balance-method round_robin \
#   --disaggregation-transfer-backend nixl &

# echo "Starting SGLang Decode Server (port 30001) on GPU1..."
# python -m sglang.launch_server \
#   --model-path $MODEL \
#   --disaggregation-mode decode \
#   --tensor-parallel-size 1 \
#   --data-parallel-size 2 \
#   --base-gpu-id 2 \
#   --chunked-prefill-size -1 \
#   --port 30002 \
#   --disable-cuda-graph \
#   --mem-fraction-static 0.7 \
#   --max-total-tokens 10000 \
#   --prefill-round-robin-balance 1,1 \
#   --disaggregation-transfer-backend nixl  &

echo "Starting Router (port 8000)..."
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30002 \
  --host 0.0.0.0 \
  --port 8000 &
