from CAT.dataset import AdapTestDataset
from CAT.strategy.abstract_strategy import AbstractStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import csv
from typing import List



class LLMTest(AbstractStrategy):

    def __init__(self):
        model_name = "/d2/llms/Qwen2.5-1.5B-Instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',truncation_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",use_cache=False,device_map="auto")
        # self.model.to(self.device)
        self.batch_size = 1
        super().__init__()

    @property
    def name(self):
        return 'LLMTest'
    
    def load_id_maps(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            id_maps = json.load(f)
        return id_maps

    def get_original_id(self,mapped_id, id_maps):
        student_map = id_maps.get('test_student_map', {})
        original_ids = student_map.get('original', [])
        mapped_ids = student_map.get('mapped', [])
        try:
            index = mapped_ids.index(mapped_id)
            return original_ids[index]
        except (ValueError, IndexError):
            return None
        
    def get_mapped_id(self, original_id, id_maps):
        question_map = id_maps.get('question_map', {})
        original_ids = question_map.get('original', [])
        mapped_ids = question_map.get('mapped', [])
        try:
            original_id = str(original_id)
            index = original_ids.index(original_id)
            return mapped_ids[index]
        except (ValueError, IndexError):
            return None
        
    def get_original_question_id(self, mapped_id, id_maps):
        question_map = id_maps.get('question_map', {})
        original_ids = question_map.get('original', [])
        mapped_ids = question_map.get('mapped', [])
        try:
            index = mapped_ids.index(mapped_id)
            return original_ids[index]
        except (ValueError, IndexError):
            return None
    
    def extract_response(self, response):
        pattern = r'\bPm_\d+\b'
        match = re.search(pattern, response)
        if match:
            return match.group()
        else:
            return None
        
    def judge_selection(self, response):
        pattern = r'<select>(.*?)</select>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return content
        else:
            return None
        
    def contains_string(self, lst, target_string):
        if not isinstance(lst, list):
            raise TypeError("输入的变量必须是列表类型")
        for item in lst:
            if isinstance(item, dict) and target_string in item:
                return True
        return False
    
    def extract_is_correct(self, data, my_dict):
        results = {}
        for student_id, problem_ids in my_dict.items():
            results[student_id] = []
            for problem_id in problem_ids:
                for row in data:
                    if int(row['student_id']) == student_id and int(row['question_id']) == problem_id:
                        results[student_id].append(row['correct'])
                        break  
        return results
    
    @staticmethod
    def print_memory_usage(prefix=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  
            reserved = torch.cuda.memory_reserved() / 1024**3    
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  
            print(f"{prefix} | Current: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")
        else:
            print("CUDA not available, memory tracking disabled")

    def cat_infer(self, test_data, theta,student_id):
        question_bank = json.load(open('datasets/MOOCRadar/question_bank.json', 'r'))
        id_maps_file = 'datasets/MOOCRadar/id_maps.json'
        id_maps = self.load_id_maps(id_maps_file)
        data = test_data.tested
        csv_file_path = 'datasets/MOOCRadar/test_triples.csv'
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            csv_data = list(reader)
        test_results = self.extract_is_correct(csv_data, data)
        selected_questions = {}
        batch_data = []
        student_ids = []
        for s_id in range(test_data.num_students):
            str_id = self.get_original_id(student_id, id_maps)
            str_id = 'U_' + str_id
            if str_id in question_bank:
                student_questions = question_bank[str_id]
                s_q = data[0]
                    
                sq_c = [] # 获取当前学生所选题题目的内容
                for q in s_q:
                    q_o_id = self.get_original_question_id(q, id_maps)
                    q_o_id = 'Pm_' + q_o_id
                    for student_question in student_questions:
                        if q_o_id in student_question:
                            q_c = student_question[q_o_id]
                            sq_c.append(q_c)
                    
                history_questions = [] # 构造当前学生的历史答题记录（问题内容，正确与否）
                i = 0
                for q in sq_c:
                    q_content = q['content']
                    history_correction = test_results[0][i]
                    q_all_content = q_content + (' (correct)' if history_correction == '1' else ' (incorrect)')
                    history_questions.append(q_all_content)
                    i += 1
                    
                untested_questions = [] # 获取当前学生的未答题记录  
                original_untested_questions = test_data.untested[0]
                for q in original_untested_questions:
                    q_o_id = self.get_original_question_id(q, id_maps)
                    q_o_id = 'Pm_' + q_o_id
                    for student_question in student_questions: # 获取未选题的内容
                        if q_o_id in student_question:
                            untested_questions.append(student_question)

                batch_data.append((theta, history_questions, untested_questions))
                student_ids.append(s_id)
            
        responses = self.batch_get_chat([(theta, h, u) for theta, h, u in batch_data])
        return responses[0]
    
    def adaptest_select(self, test_data, theta,student_id):
        pass
        # selected_questions = {}
        # for s_id, response in zip(student_ids, responses):
        #     selection = self.judge_selection(response)
        #     if selection is not None: # 目前处于选题动作
        #         ex_selection = self.extract_response(selection)
        #         if ex_selection is not None:
        #             if self.contains_string(batch_data[student_ids.index(s_id)][2], ex_selection): # 只有满足此条件才出发选题逻辑
        #                 print(ex_selection)
        #                 ex_response = ex_response.replace('Pm_', '')
        #                 s_question = int(ex_response)
        #                 mapped_question = self.get_mapped_id(s_question, id_maps)
        #                 selected_questions[s_id] = mapped_question
        #                 return selected_questions
        #             else:
        #                 print('selection wrong question')
        #         else:
        #             print('selection format error')
        #     else: # 后续添加answer逻辑
        #         print('no selection')
        # return selected_questions

    def batch_get_chat(self, batch: List[tuple]) -> List[str]:
        all_responses = []
        for i in range(0, len(batch), self.batch_size):  # 分块处理
            chunk = batch[i:i+self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1} of {len(batch)//self.batch_size}")
            prompts = []
            for theta, history, untested in chunk:
                prompt = f"""
                Your task is to simulate the entire process of a computer-adaptive test. 
                You can select questions for the student based on their current answers, the cognitive diagnostic model’s assessment of the student’s ability, and the questions they have already selected. 
                You can also decide whether the student still needs further assessment.
                You must first reason within <think>...</think>. 
                If you believe the student still needs more questions to be assessed more accurately, you can output a selected question for the student within <select>...</select> after the <think>...</think> part.
                Remember, you can only select one question at a time.
                When you believe that the student no longer requires assessment, you can output the final evaluation report for the student within <answer>...</answer>.
                student's ability: {theta}
                candidate questions: {untested}
                history questions: {history}
                Output format for selecting specific questions: 
                <think>
                ... 
                </think> 
                <select>
                ...
                </select>
                Output format for the final evaluation report: 
                <think>
                ... 
                </think> 
                <answer>
                ... 
                </answer>
                """
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(text)
            inputs = self.tokenizer(
                prompts, 
                padding=True, 
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            outputs = outputs.cpu()
            torch.cuda.empty_cache()
            all_responses.extend(
                self.tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
            )
        return all_responses