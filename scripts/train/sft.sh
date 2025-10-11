export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES='4,5,6,7' accelerate launch /home/hmpiao/adv_reason/Reasoning360/sft.py