"""
Tool generation manager for LLM agents
"""

import torch
import re
import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import random

from .tensor_helper import TensorHelper, TensorConfig
from agent.tool.tool_env import ToolEnv, step, step_batch
from verl import DataProto
from verl.utils.tracking import Tracking

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_tool_response_length: int  # Renamed from max_obs_length
    num_gpus: int
    # use_parallel_tool_calls: bool = False
    use_batch_tool_calls: bool = False  # New option for batch execution
    tool_call_start: str = "<select>"
    tool_call_end: str = "</select>"
    tool_response_start: str = "<knowledge>"
    tool_response_end: str = "</knowledge>"
    tool_custom_response_template: str = ""
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,  # Renamed
            max_start_length=config.max_start_length,
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _process_tool_call(self, responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            tool_pattern = r'<select>(.*?)</select>'
            match = re.search(tool_pattern, resp, re.DOTALL)
            
            if not match:
                return resp + self.tokenizer.eos_token, False  # No tool call found
            
            resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end
            
            return resp + self.tokenizer.eos_token, True
        
        # Process each response string
        return [process_single_response(resp)[0] for resp in responses_str], [process_single_response(resp)[1] for resp in responses_str]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        # Extract the first tool call from each response
        responses_str, active_masks = self._process_tool_call(responses_str) # 分别返回调用工具前的请求以及是否调用工具
        # Tokenize processed responses
        cleaned_token_ids = self._batch_tokenize(responses_str)
        
        return cleaned_token_ids, responses_str, torch.tensor(active_masks, dtype=torch.bool)
    
    def _process_tool_responses(self, tool_responses: List[str]) -> torch.Tensor: # 对工具响应进行tokenize并限制长度
        """Process tool responses to token ids"""
        
        tool_responses_ids = self.tokenizer(
            tool_responses, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if tool_responses_ids.shape[1] > self.config.max_tool_response_length:
            print("[WARNING] TOOL RESPONSE TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            tool_responses_ids = tool_responses_ids[:, :self.config.max_tool_response_length]
            
        return tool_responses_ids
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv], 
                          active_mask: torch.Tensor,
                          student_ids: List[str]) -> List[str]:
        """Execute tool calls sequentially and return tool responses."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env, active) in enumerate(zip(response_strs, envs, active_list)):
            if not active:
                continue
                
            # Step the environment using the agent's response
            result = step(env, resp,student_ids[i])
            tool_response = result[0]  # Extract observation from (observation, reward, done, info)
            tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
        return tool_responses
    
    def _execute_tool_calls_batch(self, response_strs: List[str], 
                                 envs: List[ToolEnv], 
                                 active_mask: torch.Tensor,
                                 student_ids: List[str],
                                 num_samples: int,
                                 is_reset: bool,
                                 step: int,
                                 count: int):
        """Execute tool calls in batch for tools that support batch operations."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        is_empty = False
        # Filter active environments and responses
        active_envs = []
        active_responses = []
        active_indices = []
        active_student_ids = []
        for i, (env, resp, active) in enumerate(zip(envs, response_strs, active_list)):
            if active:
                active_envs.append(env)
                active_responses.append(resp)
                active_indices.append(i)
                active_student_ids.append(student_ids[i])
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs) # 初始化每个学生的工具响应数据
        print("active student ids:",active_student_ids)
        if not active_envs:
            return tool_responses
            
        # Use the independent step_batch function for active environments
        batch_results, select_final_questions, accs = step_batch(active_envs, active_responses,active_student_ids,num_samples,is_reset,step,count) # 返回当前batch的学生选题结果、选题库和准确率
        print("batch_results: ", batch_results)
        print("select_final_questions: ", select_final_questions)
        # Map results back to original indices
        for idx, result in zip(active_indices, batch_results):
            if result is None:
                tool_responses[idx] = ""
            else:
                tool_response = result[0]  # Extract observation from (observation, reward, done, info)
                tool_responses[idx] = self.config.tool_custom_response_template.format(tool_response=tool_response)
        return tool_responses, select_final_questions, accs
    
    # def _update_rolling_state(self, rollings, cur_responses: torch.Tensor,
    #                     tool_responses_ids: torch.Tensor, step: int) -> DataProto:
        
    #     device = cur_responses.device  
    #     prefix_prompt = """
    #             Your task is to simulate the entire process of a computer-adaptive test. 
    #             You can select questions for the student based on their current answers, the cognitive diagnostic model's assessment of the student's ability, and the questions they have already selected. 
    #             You can also decide whether the student still needs further assessment.
    #             You must first reason within <think>...</think>. 
    #             If you believe the student still needs more questions to be assessed more accurately, you can output a selected question for the student within <select>...</select> after the <think>...</think> part.
    #             Remember, you can only select one question at a time.
    #             When you believe that the student no longer requires assessment, you can output the final evaluation report for the student within <answer>...</answer>.
    #             Output format for selecting specific questions: 
    #             <think>
    #             ... 
    #             </think> 
    #             <select>
    #             ...
    #             </select>
    #             Output format for the final evaluation report: 
    #             <think>
    #             ... 
    #             </think> 
    #             <answer>
    #             ... 
    #             </answer>
    #         """
    #     batch_size = cur_responses.size(0)
    #     prefix_prompts = [prefix_prompt] * batch_size # 扩展到batch_size维度
    #     # prefix_ids = self.tokenizer(
    #     #     prefix_prompt,
    #     #     return_tensors="pt",
    #     #     padding="max_length",
    #     #     truncation=True,
    #     #     max_length=self.config.max_prompt_length
    #     # ).input_ids.to(device)
    #     prefix_ids = self._batch_tokenize(prefix_prompts) 
        
    #     new_input_ids = self.tensor_fn.concatenate_with_padding([
    #         prefix_ids,
    #         cur_responses,
    #         tool_responses_ids
    #     ])
        
    #     new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
    #     new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        
    #     effective_len = new_attention_mask.sum(dim=1).max()
    #     max_len = min(self.config.max_prompt_length, effective_len)
        
    #     return DataProto.from_dict({
    #         'input_ids': new_input_ids[:, -max_len:],
    #         'position_ids': new_position_ids[:, -max_len:],
    #         'attention_mask': new_attention_mask[:, -max_len:]
    #     })
    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            tool_responses_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            tool_responses_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
    
    def _update_right_side(self, right_side: Dict,  # 更新right_side状态，将响应与工具响应拼接
                          cur_responses: torch.Tensor,
                          tool_responses_ids: torch.Tensor) -> Dict:
        """Update right side state."""
        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            cur_responses,
            tool_responses_ids
        ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}

    def _extract_student_id(self, prompt: str) -> Optional[str]:
        match = re.search(r"student's ID:\s*(\d+)", prompt)
        return match.group(1) if match else None

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
            
        padded_active_batch = DataProto.from_dict(padded_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def _run_validate_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None,num_samples:int = 0,is_reset:bool = False) -> Tuple[Dict, Dict]: # 验证集生成逻辑
        print("is validate")
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        batch_size = gen_batch.batch['input_ids'].shape[0]
        input_ids = self.tokenizer.batch_decode(gen_batch.batch['input_ids'], skip_special_tokens=True)
        student_ids = [self._extract_student_id(prompt) for prompt in input_ids] # 获取数据集里面的学生id
        active_mask = torch.ones(batch_size, dtype=torch.bool) # 初始全部激活
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch # 初始化输入
        select_final_questions = None
        total_accs = [None] * num_samples # 初始化用于存储每个学生的准确率的数组
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses']) # 截止至第一次调用工具
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1
            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_responses, select_questions, accs = self._execute_tool_calls_batch(responses_str, envs, active_mask,student_ids,num_samples,is_reset)[:3] # 获取每一轮工具响应以及最后一轮的所有选题以及准确率，此处默认获取最后一次正确调用工具的准确率
                is_reset = False # 确保同一batch第二轮不会重置
                print("tmp_accs: ", accs) # 假设为长度为16的数组
                for i in range(len(accs)):
                    if accs[i] is not None:
                        total_accs[i] = accs[i] # 非空再对当前的学生进行更新
                if select_questions != {}: # 防止选题为空
                    select_final_questions = select_questions
            else:
                # Use sequential execution for tool calls
                tool_responses = self._execute_tool_calls(responses_str, envs, active_mask,student_ids) # 获取工具响应
                

            active_num_list.append(active_mask.sum().item())
            tool_responses_ids = self._process_tool_responses(tool_responses)
            # Update states
            rollings = self._update_rolling_state(
                rollings, 
                responses_ids,
                tool_responses_ids
            )
            original_right_side = self._update_right_side( # 记录每一轮的responses和工具调用结果
                original_right_side,
                responses_ids,
                tool_responses_ids
            )
            torch.cuda.empty_cache()

        print("ACTIVE_TRAJ_NUM:", active_num_list)
        original_right_side['turns'] = turns
        for i in range(len(total_accs)):
            if total_accs[i] is None:
                total_accs[i] = 0
        # Save trajectory and return final output
        return self._compose_final_output(original_left_side, original_right_side, meta_info), select_final_questions, total_accs
    
    def _merge_group_outputs(self, group_outputs):
        # 如果输入是DataProto对象，需要先转换为字典
        print("len of group_outputs: ", len(group_outputs))
        if hasattr(group_outputs[0], '__dict__'):
            merged_output = {}
            for key in group_outputs[0].__dict__.keys():
                if isinstance(group_outputs[0].__dict__[key], torch.Tensor):
                    merged_output[key] = torch.cat([output.__dict__[key] for output in group_outputs])
                else:
                    merged_output[key] = group_outputs[0].__dict__[key]
        else:
            # 原来的处理逻辑
            merged_output = {}
            for key in group_outputs[0].keys():
                if isinstance(group_outputs[0][key], torch.Tensor):
                    merged_output[key] = torch.cat([output[key] for output in group_outputs])
                else:
                    merged_output[key] = group_outputs[0][key]
        return merged_output

    def _run_training_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None, num_samples:int = 5, is_reset:bool = False,count:int = 0):
        print("is training")
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        batch_size = gen_batch.batch['input_ids'].shape[0]
        input_ids = self.tokenizer.batch_decode(gen_batch.batch['input_ids'], skip_special_tokens=True)
        student_ids = [self._extract_student_id(prompt) for prompt in input_ids] # 获取数据集里面的学生id
        unique_instance_ids = []
        id_counters = defaultdict(int)
        for s_id in student_ids:
            unique_id = f"{s_id}_{id_counters[s_id]}"
            unique_instance_ids.append(unique_id)
            id_counters[s_id] += 1
        active_mask = torch.ones(batch_size, dtype=torch.bool) # 初始全部激活
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch # 初始化输入
        select_final_questions = None
        total_accs = [None] * 80 # 初始化用于存储每个学生的准确率的数组
        # Main generation loop
        for step in range(self.config.max_turns): # 将第一轮作为初始轮
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses']) # 截止至第一次调用工具
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1
            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_responses, select_questions, accs = self._execute_tool_calls_batch(responses_str, envs, active_mask,unique_instance_ids,num_samples,is_reset,step,count)[:3] # 获取每一轮工具响应以及最后一轮的所有选题以及准确率，此处默认获取最后一次正确调用工具的准确率
                is_reset = False # 确保同一batch第二轮不会重置
                print("tmp_accs: ", accs) 
                for i in range(len(accs)):
                    if accs[i] is not None:
                        total_accs[i] = accs[i] # 非空再对当前的学生进行更新
                if select_questions != {}: # 防止选题为空
                    select_final_questions = select_questions
            else:
                # Use sequential execution for tool calls
                tool_responses = self._execute_tool_calls(responses_str, envs, active_mask,student_ids) # 获取工具响应
                

            active_num_list.append(active_mask.sum().item())
            tool_responses_ids = self._process_tool_responses(tool_responses)
            # Update states
            rollings = self._update_rolling_state(
                rollings, 
                responses_ids,
                tool_responses_ids
            )
            original_right_side = self._update_right_side( # 记录每一轮的responses和工具调用结果
                original_right_side,
                responses_ids,
                tool_responses_ids
            )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        original_right_side['turns'] = turns
        for i in range(len(total_accs)):
            if total_accs[i] is None:
                total_accs[i] = 0
        # Save trajectory and return final output
        return self._compose_final_output(original_left_side, original_right_side, meta_info), select_final_questions, total_accs

    def run_llm_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None,is_validate:bool = False,num_samples:int = 0,is_reset:bool = False,count:int = 0) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        if is_validate:
            return self._run_validate_loop(gen_batch, envs, initial_input_ids,num_samples,is_reset)
        else:
            return self._run_training_loop(gen_batch, envs, initial_input_ids,num_samples,is_reset,count)
        
        



    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        if right_side['responses'].dtype != torch.int64: # 转换成int类型
            right_side['responses'] = right_side['responses'].to(torch.int64)
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        

        # Print or return the text content
        # print("Prompts Text:", prompts_text[:20])
        # print("Input IDs Text:", input_ids_text[:20])
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output