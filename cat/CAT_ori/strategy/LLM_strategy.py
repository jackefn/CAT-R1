from CAT.dataset import AdapTestDataset
from CAT.strategy.abstract_strategy import AbstractStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import csv
from typing import List



class LLM(AbstractStrategy):

    def __init__(self):
        model_name = "/d2/llms/Qwen2.5-1.5B-Instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left',truncation_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",use_cache=False,device_map="auto")
        # self.model.to(self.device)
        self.batch_size = 16  
        super().__init__()

    @property
    def name(self):
        return 'LLM'
    
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

    def adaptest_select(self, test_data, thetas, it):
        question_bank = json.load(open('datasets/MOOCRadar/question_bank.json', 'r'))
        id_maps_file = 'datasets/MOOCRadar/id_maps.json'
        id_maps = self.load_id_maps(id_maps_file)
        if it == 1:
            selected_questions = {}
            batch_data = []
            student_ids = []
            for s_id in range(test_data.num_students):
                str_id = self.get_original_id(s_id, id_maps) # 获取原始学生ID
                str_id = 'U_' + str_id
                if str_id in question_bank:
                    student_questions = question_bank[str_id] # 获取学生对应的初始题目（题库里的题目）
                    theta = thetas[s_id]
                    batch_data.append((theta, student_questions))
                    student_ids.append(s_id)

            responses = self.batch_get_chat_first([(theta, q) for theta, q in batch_data]) # 传入所有学生
            for s_id, response in zip(student_ids, responses):
                ex_response = self.extract_response(response)
                if ex_response is not None:
                    if self.contains_string(batch_data[student_ids.index(s_id)][1], ex_response):
                        ex_response = ex_response.replace('Pm_', '')
                        s_question = int(ex_response)
                        mapped_question = self.get_mapped_id(s_question, id_maps)
                        selected_questions[s_id] = mapped_question
                    else:
                        response = self.extract_response(str(batch_data[student_ids.index(s_id)][1]))
                        ex_response = response.replace('Pm_', '')
                        s_question = int(ex_response)
                        mapped_question = self.get_mapped_id(s_question, id_maps)
                        selected_questions[s_id] = mapped_question
                else:
                    response = self.extract_response(str(batch_data[student_ids.index(s_id)][1]))
                    ex_response = response.replace('Pm_', '')
                    s_question = int(ex_response)
                    mapped_question = self.get_mapped_id(s_question, id_maps)
                    selected_questions[s_id] = mapped_question
            
            return selected_questions
        else:
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
                str_id = self.get_original_id(s_id, id_maps)
                str_id = 'U_' + str_id
                if str_id in question_bank:
                    student_questions = question_bank[str_id]
                    theta = thetas[s_id]
                    s_q = data[s_id]
                    
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
                        history_correction = test_results[s_id][i]
                        q_all_content = q_content + (' (correct)' if history_correction == '1' else ' (incorrect)')
                        history_questions.append(q_all_content)
                        i += 1
                    
                    untested_questions = [] # 获取当前学生的未答题记录  
                    original_untested_questions = test_data.untested[s_id]
                    for q in original_untested_questions:
                        q_o_id = self.get_original_question_id(q, id_maps)
                        q_o_id = 'Pm_' + q_o_id
                        for student_question in student_questions: # 获取未选题的内容
                            if q_o_id in student_question:
                                untested_questions.append(student_question)

                    batch_data.append((theta, history_questions, untested_questions))
                    student_ids.append(s_id)
            
            responses = self.batch_get_chat_second([(theta, h, u) for theta, h, u in batch_data])
            
            selected_questions = {}
            for s_id, response in zip(student_ids, responses):
                ex_response = self.extract_response(response)
                if ex_response is not None:
                    if self.contains_string(batch_data[student_ids.index(s_id)][2], ex_response):
                        ex_response = ex_response.replace('Pm_', '')
                        s_question = int(ex_response)
                        mapped_question = self.get_mapped_id(s_question, id_maps)
                        selected_questions[s_id] = mapped_question
                    else:
                        response = self.extract_response(str(batch_data[student_ids.index(s_id)][2]))
                        ex_response = response.replace('Pm_', '')
                        s_question = int(ex_response)
                        mapped_question = self.get_mapped_id(s_question, id_maps)
                        selected_questions[s_id] = mapped_question
                else:
                    response = self.extract_response(str(batch_data[student_ids.index(s_id)][2]))
                    ex_response = response.replace('Pm_', '')
                    s_question = int(ex_response)
                    mapped_question = self.get_mapped_id(s_question, id_maps)
                    selected_questions[s_id] = mapped_question
            
            return selected_questions

    def batch_get_chat_first(self, batch: List[tuple]) -> List[str]:
        all_responses = []
        for i in range(0, len(batch), self.batch_size):  # 分块处理
            print(f"Processing batch {i//self.batch_size + 1} of {len(batch)//self.batch_size}")
            self.print_memory_usage(f"Before batch {i//self.batch_size + 1}")
            chunk = batch[i:i+self.batch_size]
            
            prompts = []
            for theta, question_bank in chunk:
        
                prompt = f"""
                By thinking step by step about the student's strengths and weaknesses, select the most suitable question from the
                candidate questions, considering difficulty, discrimination, and knowledge coverage. Use the principles of
                computerized adaptive testing so that after the student answers this question, their ability can be measured more
                accurately. In your reasoning process, you need to think carefully.
                student's ability: {theta}
                candidate questions: {question_bank}

                Based on this analysis, consider the following characteristics when evaluating candidate questions:
                Difficulty:
                Ensure the question is appropriately challenging, providing a balance between too easy and too difficult.

                Discrimination:
                Choose questions that effectively differentiate between varying levels of student ability.

                Knowledge Coverage:
                Select questions that cover key knowledge areas necessary for the student to reinforce or master.

                Incorporate the principles of computerized adaptive testing (CAT):
                Dynamic Adjustment:
                Adjust the difficulty of questions based on the student's real−time performance to assess their ability accurately.

                Precise Measurement:
                Ensure each question helps in narrowing the estimation of the student's ability, increasing the accuracy of the
                assessment.

                Emphasize the reasoning behind your selection, providing a clear and logical explanation:
                Explain the Selection:
                Clearly state why you selected the particular question from the candidate list.
                Detail how this question addresses the student's current strengths and weaknesses.
                Explain how the question will help in detecting and measuring the student's cognitive state and ability.

                Now start selecting and organizing your output by strictly following the output format below:
                Reason: Explain why you select the question, how to detect cognitive state with the question
                Selected question with index: the selected question here with the index, following the output format like <the index>,
                only one index here
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
    
            # 批量处理
            inputs = self.tokenizer(
                prompts, 
                padding=True, 
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            self.print_memory_usage("After tokenizer")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            outputs = outputs.cpu()
            torch.cuda.empty_cache()
            self.print_memory_usage("After generate")
            all_responses.extend(
                self.tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
            )
        return all_responses

    def batch_get_chat_second(self, batch: List[tuple]) -> List[str]:
        all_responses = []
        for i in range(0, len(batch), self.batch_size):  # 分块处理
            chunk = batch[i:i+self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1} of {len(batch)//self.batch_size}")
            prompts = []
            for theta, history, untested in chunk:
                prompt = f"""
                By thinking step by step about the student's strengths and weaknesses, select the most suitable question from the
                candidate questions, considering difficulty, discrimination, and knowledge coverage. Use the principles of
                computerized adaptive testing so that after the student answers this question, their ability can be measured more
                accurately. In your reasoning process, you need to think carefully.

                student's ability: {theta}
                candidate questions: {untested}
                history questions: {history}

                Remember: You must select one question from the candidate questions, not the history questions!

                Based on this analysis, consider the following characteristics when evaluating candidate questions:
                Difficulty:
                Ensure the question is appropriately challenging, providing a balance between too easy and too difficult.

                Discrimination:
                Choose questions that effectively differentiate between varying levels of student ability.

                Knowledge Coverage:
                Select questions that cover key knowledge areas necessary for the student to reinforce or master.

                Incorporate the principles of computerized adaptive testing (CAT):
                Dynamic Adjustment:
                Adjust the difficulty of questions based on the student's real−time performance to assess their ability accurately.

                Precise Measurement:
                Ensure each question helps in narrowing the estimation of the student's ability, increasing the accuracy of the
                assessment.

                Emphasize the reasoning behind your selection, providing a clear and logical explanation:
                Explain the Selection:
                Clearly state why you selected the particular question from the candidate list.
                Detail how this question addresses the student's current strengths and weaknesses.
                Explain how the question will help in detecting and measuring the student's cognitive state and ability.

                Now start selecting and organizing your output by strictly following the output format below:
                Reason: Explain why you select the question, how to detect cognitive state with the question
                Selected question with index: the selected question here with the index, following the output format like <the index>,
                only one index here
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
            self.print_memory_usage("After chat template")
            inputs = self.tokenizer(
                prompts, 
                padding=True, 
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            self.print_memory_usage("After tokenizer")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.eos_token_id
            )
            outputs = outputs.cpu()
            torch.cuda.empty_cache()
            self.print_memory_usage("After generate")
            all_responses.extend(
                self.tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
            )
        return all_responses