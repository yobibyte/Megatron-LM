#!/bin/bash 

TP=${TP:-2}
PP=${PP:-1}
EP=${EP:-4}
NODES_REQUIRED=${NODES_REQUIRED:-1}

echo "Using Qwen3-30B-A3B model checkpoint"
SCRIPT_PATH="${BASH_SOURCE[0]}"
source $(dirname $SCRIPT_PATH)/common.sh

# Default values
GRPO_CLAMP_EPS_LOWER=${GRPO_CLAMP_EPS_LOWER:-0.2}
GRPO_CLAMP_EPS_UPPER=${GRPO_CLAMP_EPS_UPPER:-0.28}
MAX_INFERENCE_BS=${MAX_INFERENCE_BS:-32}
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-16}
GRPO_PROMPTS_PER_STEP=${GRPO_PROMPTS_PER_STEP:-64}
GRPO_ITERATIONS=${GRPO_ITERATIONS:-1}
GRPO_KL_BETA=${GRPO_KL_BETA:-"0.0"}
TRAINING_BATCH_SIZE=${TRAINING_BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-8192}
EXIT_INTERVAL=${EXIT_INTERVAL:-20}
CHKPT_SAVE_INTERVAL=${CHKPT_SAVE_INTERVAL:-20}

ENV_DEPENDENT="\
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $TRAINING_BATCH_SIZE \
--grpo-group-size $GRPO_GROUP_SIZE \
--grpo-prompts-per-step $GRPO_PROMPTS_PER_STEP \
--grpo-iterations $GRPO_ITERATIONS \
--grpo-clamp-eps-lower $GRPO_CLAMP_EPS_LOWER \
--grpo-clamp-eps-upper $GRPO_CLAMP_EPS_UPPER \
--grpo-kl-beta $GRPO_KL_BETA \
--langrl-env-config $ENV_CONFIG \
--seq-length $MAX_SEQ_LENGTH \
--inference-max-seq-length $MAX_SEQ_LENGTH \
--inference-max-requests $MAX_INFERENCE_BS \
--pretrained-checkpoint $CHECKPOINT \
--rl-skip-bos-token \
--rl-default-top-k -1 \
--rl-default-temperature 1.0 \
--rl-default-top-p 1.0 \
--rl-importance-sampling-truncation-coef 10.0 \
--rl-inference-logprobs-is-correction \
--no-rl-use-sequence-packing \
--moe-pad-experts-for-cuda-graph-inference \
--no-use-tokenizer-model-from-checkpoint-args \
--moe-pad-experts-for-cuda-graph-inference \
--inference-dynamic-batching-max-tokens 8192 \
--inference-dynamic-batching-max-requests 128 \
--inference-dynamic-batching-num-cuda-graphs 2 \
--decode-only-cuda-graphs \
--cuda-graph-impl local \
--cuda-graph-scope full \
--seq-length $MAX_SEQ_LENGTH \
--inference-max-seq-length $MAX_SEQ_LENGTH \
--bf16 \
--tensor-model-parallel-size $TP  \
--pipeline-model-parallel-size $PP  \
--expert-model-parallel-size $EP \
--expert-tensor-parallel-size 1 \
--attention-backend flash \
--transformer-impl transformer_engine \
--te-rng-tracker \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model Qwen/Qwen3-30B-A3B \
--tokenizer-hf-include-special-tokens \
--untie-embeddings-and-output-weights \
--num-layers 48 \
--hidden-size 2048 \
--ffn-hidden-size 6144 \
--num-attention-heads 32 \
--kv-channels 128 \
--max-position-embeddings $MAX_SEQ_LENGTH \
--group-query-attention \
--num-query-groups 4 \
--normalization RMSNorm \
--norm-epsilon 1e-6 \
--position-embedding-type rope \
--rotary-percent 1.0 \
--rotary-base 1000000 \
--use-rotary-position-embeddings \
--swiglu \
--disable-bias-linear \
--num-experts 128 \
--moe-router-dtype fp64 \
--moe-router-topk 8 \
--moe-ffn-hidden-size 768 \
--moe-aux-loss-coeff 0.0 \
--moe-router-load-balancing-type aux_loss \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 \
--vocab-size 151936 \
--make-vocab-size-divisible-by 128 \
--dist-ckpt-strictness log_unexpected \
--qk-layernorm \
--moe-token-dispatcher-type alltoall \
--moe-layer-freq 1 \
--optimizer adam \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--adam-eps 1e-8 \
--lr 3e-6 \
--min-lr 3e-6 \
--init-method-std 0.014 \
--lr-decay-style constant \
--lr-warmup-samples 640 \
--lr-warmup-init 0.3e-7 \
--clip-grad 1.0 \
--weight-decay 0.01 \
--no-load-optim \
--ckpt-format torch_dist "
