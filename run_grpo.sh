while getopts "p:m:d:" opt; do
  case $opt in
    p) path=$OPTARG ;;
    m) model=$OPTARG ;;
    d) dataset=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

shift $((OPTIND - 1))

export VLLM_ATTENTION_BACKEND=XFORMERS
export BASE_MODEL="${path}"

# export PROJECT_NAME='CAT-R1-MOOCRadar'
# export EXPERIMENT_NAME="${model}_${dataset}_grpo"
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2,4

# export WANDB_API_KEY="e74ad1b52419ca7191227f6ac05268cfb3547de"
export HF_ENDPOINT="https://hf-mirror.com"
set -x

# ray stop
ray start --head --port 6832

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=datasets/MOOCRadar/trained_data_0_639.parquet \
    data.val_files=datasets/MOOCRadar/trained_data_0_639.parquet \
    data.train_batch_size=16 \
    data.shuffle=False \
    data.max_prompt_length=32000 \
    data.max_response_length=8192 \
    data.max_start_length=32000 \
    data.max_tool_response_length=8192 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n_repeat=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    tool.env='select' $@