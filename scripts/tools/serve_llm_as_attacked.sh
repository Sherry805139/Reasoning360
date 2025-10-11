SLURM_NNODES=1
SLURM_CPUS_PER_TASK=48
export NCCL_P2P_DISABLE=1 

#gpu_ids=2,3
#export CUDA_VISIBLE_DEVICES=${gpu_ids}

# (1) detect this node’s primary IP
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Detected NODE_IP = $NODE_IP"

# (2) export judge URL for downstream clients
export STEM_LLM_JUDGE_URL="http://${NODE_IP}:8000"
echo "STEM_LLM_JUDGE_URL=$STEM_LLM_JUDGE_URL"

# (3) launch the vLLM server bound to that IP
CUDA_VISIBLE_DEVICES='0' vllm serve /home/hmpiao/adv_reason/Reasoning360/checkpoints/Reasoning360-1.7B/--Qwen3-1.7B-Base-think-qwen2chat-sftjudge-ke-1-e6-s1-directcotstepjudge-frompretrain-0.2/global_step_40/actor/Qwen3-1.7B-Base-think-qwen2chat-sftjudge-ke-1-e6-s1-step40-directcotstepjudge-frompretrain-0.2 --host "$NODE_IP" --data-parallel-size 1 --tensor-parallel-size 1 --gpu-memory-utilization 0.8 --max-model-len 7168
