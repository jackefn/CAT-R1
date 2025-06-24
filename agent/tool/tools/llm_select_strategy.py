import json
import requests
import random
from typing import List, Dict, Tuple
import select_question
from vllm import LLM, SamplingParams

class LLMSelectStrategy:
    def __init__(self, llm_api_url: str = "http://localhost:8006"):
        self.llm_api_url = llm_api_url
        self.llm = LLM(
            model="/d2/mxy/Models/Qwen2-7B",
            tokenizer="/d2/mxy/Models/Qwen2-7B",
            tensor_parallel_size=1,
            max_model_len=4096,
            enforce_eager=True,
            gpu_memory_utilization=0.8
        )
        self.base_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=2048,
            skip_special_tokens=True
        )

    def build_chat_template(self, messages: List[dict]) -> str:
        template = ""
        for msg in messages:
            if msg["role"] == "system":
                template += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                template += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        template += "<|im_start|>assistant\n"
        return template

    def build_select_prompt(self, history_questions: List[Dict], untested_questions: List[Dict], student_id: str) -> str:
        prompt = f"""Based on the student's (ID: {student_id}) answer history and available untested questions, select the most appropriate next question.

        Answer History:
        {json.dumps(history_questions, indent=2, ensure_ascii=False)}

        Available Untested Questions:
        {json.dumps(untested_questions, indent=2, ensure_ascii=False)}

        Please analyze:
        1. The student's current knowledge level and learning progress
        2. The difficulty level and knowledge points of each untested question
        3. The most suitable next question that will help assess and improve the student's learning

        Select one question ID from the untested questions that you think is most appropriate for this student.
        You should only answer with the question ID (e.g., "Pm_123"), don't add any other words.
        """
        messages = [
            {"role": "system", "content": "You are an expert educational assessment specialist."},
            {"role": "user", "content": prompt}
        ]
        return self.build_chat_template(messages)

    def batch_select_questions(self, test_data, student_ids: List[str], id_maps: Dict, question_bank: Dict, test_results: Dict) -> Dict[str, int]:
        """
        批量为学生选择下一道题目
        """
        selected_questions = {}
        prompts_to_generate = []
        student_info = []

        # 为每个学生构建提示词
        for student_id in student_ids:
            history_questions, _ = select_question.build_answered_questions(
                test_data, student_id, id_maps, question_bank, test_results
            )
            untested_questions = select_question.build_unanswered_questions(
                test_data, student_id, id_maps, question_bank
            )
            
            prompt = self.build_select_prompt(history_questions, untested_questions, student_id)
            prompts_to_generate.append(prompt)
            student_info.append({
                "student_id": student_id,
                "untested_questions": untested_questions
            })
        print(student_info)
        try:
            # 使用vLLM进行批量生成
            outputs = self.llm.generate(prompts_to_generate, self.base_params)
            
            # 处理每个学生的选择结果
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip().split("<|im_end|>")[0].strip()
                student_id = student_info[i]["student_id"]
                untested_questions = student_info[i]["untested_questions"]
                
                # 验证LLM的选择是否有效
                if select_question.contains_string(untested_questions, response):
                    # 如果选择有效，使用LLM的选择
                    question_id = int(response.replace('Pm_', ''))
                    print("question_id: ", question_id)
                    selected_questions[student_id] = select_question.get_mapped_id(question_id, id_maps)
                else:
                    # 如果选择无效，随机选择一个未答题目
                    untested_list = list(untested_questions.keys())
                    if untested_list:
                        random_question = random.choice(untested_list)
                        question_id = int(random_question.replace('Pm_', ''))
                        selected_questions[student_id] = select_question.get_mapped_id(question_id, id_maps)

        except Exception as e:
            print(f"Error in batch question selection: {e}")
            # 如果发生错误，为所有学生随机选择题目
            for student_id in student_ids:
                untested_questions = select_question.build_unanswered_questions(
                    test_data, student_id, id_maps, question_bank
                )
                untested_list = list(untested_questions.keys())
                if untested_list:
                    random_question = random.choice(untested_list)
                    question_id = int(random_question.replace('Pm_', ''))
                    selected_questions[student_id] = select_question.get_mapped_id(question_id, id_maps)

        return selected_questions 