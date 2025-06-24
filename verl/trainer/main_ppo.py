# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from agent.tool import ToolEnv
from agent.tool.tools import _default_tools

import ray
import hydra

from verl import DataProto
from verl.utils.reward_score import _default_compute_score_format, _default_compute_score_answer_f1, _default_compute_score_answer_em, _default_compute_score_format_answer
import torch
import json
import pandas as pd
from cat import CAT
import re

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        
    def compute_score_format(self, solution_str):
        """The scoring function for format reward.

        Args:
            solution_str: the solution text
        
        """
        if solution_str is None:
            return 0.0
        
        try:
            # Perfect format match for the new structure
            # First <|im_start|>assistant should have <think> and possibly <query>
            # Then <|im_start|>tool with <knowledge> (can repeat with assistant/tool pairs)
            # Final <|im_start|>assistant with the answer and <|im_end|>
            
            # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
            assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
            print("assistant_blocks: ", assistant_blocks)
            format_reward = 0.0
            
            # If no blocks found, return 0
            if not assistant_blocks:
                return 0.0
            
            # Perfect format requires at least one assistant block and matching tool blocks if tool calls exist
            # Check first assistant block contains <think> tags
            for i, assistant_block in enumerate(assistant_blocks[:-1]):
                if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<select>') == 1 and assistant_block.count('</select>') == 1:
                    think_match = re.search(r'^<think>(.*?)</think>\n<select>(.*?)</select>$', assistant_block, re.DOTALL)
                    print("think_match: ", think_match)
                    if think_match:
                        # format_reward += 0.2 * (0.8 ** i)
                        format_reward += 0.5

            # Check the last assistant block contains <answer> tags
            if assistant_blocks:  # 确保有至少一个assistant块
                last_assistant_block = assistant_blocks[-1]
                think_answer_match = re.search(r'^<think>(.*?)</think>\n<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
                print("think_answer_match: ", think_answer_match)
                if think_answer_match:
                    format_reward += 0.5
        except Exception as e:
            print(f"[DEBUG] Error in compute_score_format: {e}")
            return 0.0
        
        return format_reward

    def __call__(self, data: DataProto, select_final_questions, accs: list):

        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        # print("select_final_questions: ", select_final_questions)

        # answer_lst_auc = [] # 存储当前批次数据的auc值
        answer_lst_acc = [] # 存储当前批次数据的acc值
        format_lst = [] # 存储当前批次数据的format值
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) # 存储当前批次数据的reward值

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            # print("reward_score: ", reward_score)
            prompt_ids = data_item.batch['prompts']
            
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            
            response_ids = data_item.batch['responses']
            
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            
            valid_response_ids = response_ids[:valid_response_length].long()
            
            
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            
            pad_token_id = self.tokenizer.pad_token_id
            
            sequences_str = sequences_str.split(self.tokenizer.decode([pad_token_id]))[0]
           
            data_source = data_item.non_tensor_batch['data_source']
            
            format_score = self.compute_score_format(sequences_str)
            format_score = min(format_score, 1.0)
            format_lst.append(format_score)
        
            
            # 更新 reward_tensor
        answer_lst_acc = accs
        for i in range(len(answer_lst_acc)):
            format_reward = format_lst[i]
            format_reward = min(format_reward, 1.0)
            if format_reward == 1.0:
                reward = -1.0 + format_reward + answer_lst_acc[i]
            else:
                reward = -1.0 + format_reward
            reward_tensor[i, valid_response_length - 1] = reward

        print("answer_lst_acc: ", answer_lst_acc)
        print("format_lst: ", format_lst)
        nonzero_indices = reward_tensor.nonzero()
        nonzero_values = reward_tensor[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        print("Non-zero values:", nonzero_values)
        return reward_tensor, answer_lst_acc, format_lst
    


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    # ray.init(runtime_env={"env_vars": {"RAY_DEBUG_POST_MORTEM": "1"}})
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score)) # compute_score未使用


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    # breakpoint()
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    tools = _default_tools(config.tool.env)
    env = ToolEnv(tools=tools, max_turns=config.tool.max_turns)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=RewardManager(tokenizer=tokenizer, num_examine=0),
                            val_reward_fn=RewardManager(tokenizer=tokenizer, num_examine=1),
                            env=env)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
