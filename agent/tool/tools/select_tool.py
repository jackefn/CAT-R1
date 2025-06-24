from agent.tool.tool_base import Tool
from typing import List, Dict
import json
import pandas as pd
from cat import CAT
from agent.tool.tools import select_question
import scipy
import requests
import copy

class SelectTool(Tool):
    """
    Tool for selecting a tool from a list of tools
    """
    def __init__(self):
        name = "select"
        description = "Select an adaptive question from a list of untested questions"
        parameters = {
            "type": "object",
            "properties": {
                "select": {
                    "type": "string",
                    "description": "The name of the tool to select"
                }
            },
            "required": ["select"]
        }
        
        self.select_questions = [] # 长度为5的学生选题字典
        for j in range(5):
            tmp_select_questions = {}
            for i in range(640): # 原始学生ID数量，每个学生有5个复制环境
                tmp_select_questions[i] = [] # 构建存放学生选题数据的字典，键为原始学生ID
            self.select_questions.append(tmp_select_questions)

        # 加载数据
        test_triplets = pd.read_csv(f'datasets/MOOCRadar/train_triples_0_639.csv', encoding='utf-8')

        concept_map = json.load(open(f'datasets/MOOCRadar/concept_map.json', 'r'))
        concept_map = {int(k):v for k,v in concept_map.items()}
        student_dfs = []
        for student_id, group in test_triplets.groupby('student_id'):
            student_df = group.copy()
            student_df['student_id'] = 0 # 临时设置，以便AdapTestDataset内部处理
            student_dfs.append(student_df)

        self.test_datas = [] # 为每个学生创建AdapTestDataset的5个副本
        self.query_datas = [] # 这个变量似乎没有被使用，可以考虑移除或明确其用途
        metadata = json.load(open(f'datasets/MOOCRadar/metadata_demo.json', 'r'))

        id_maps_file = 'datasets/MOOCRadar/id_maps.json'
        self.id_maps = select_question.load_id_maps(id_maps_file)

        self.question_bank = json.load(open('datasets/MOOCRadar/question_bank.json', 'r'))

        self.test_results = select_question.extract_question_correctness(student_dfs, self.id_maps) # 获取每个学生的所有答题结果
        self.llm_api_url = "http://localhost:8100" # LLM 服务端地址
        
        # 为每个分组创建AdapTestDataset
        for student_df in student_dfs:
            student_data = student_df.to_records(index=False)
            original_data = CAT.dataset.AdapTestDataset(
                student_data, 
                concept_map,
                1,  
                metadata['num_questions'], 
                metadata['num_concepts']
            )
            
            self.test_datas.extend([copy.deepcopy(original_data) for _ in range(5)]) # 为每个原始学生复制5个环境
        
        super().__init__(name, description, parameters)
    
    
    def execute(self, tool_name: str,student_id: str):
        """
        Execute the tool (此方法在此上下文似乎未使用，或者用于单个学生处理)
        """
        print("select tool")
        print(student_id)
        pass

    def get_student_test_data(self, combined_id: str):
        """根据复合ID获取对应的测试数据副本"""
        original_id, sample_idx = map(int, combined_id.split('_'))
        
        absolute_pos = original_id * 5 + sample_idx
        
        if absolute_pos < len(self.test_datas):
            return self.test_datas[absolute_pos]
        raise IndexError(f"Invalid student ID: {combined_id}")
    
    def batch_execute(self, tool_names: List[Dict], student_ids: List[str], num_samples: int, is_reset=False, step: int = 0, count: int = 0):
        """
        为当前的学生进行选题，并根据选题后的作答记录生成评估报告，并计算每个学生的准确性
        """
        print("step: ", step)
        print("student_ids: ", student_ids)
        
        results = []
        accs = [None] * 80 
        batch_reports_request = []
        batch_simulate_requests = []
        
        replicated_student_id_to_accs_pos_map = {} 

        for i in range(len(tool_names)): 
            new_student_id = student_ids[i] 
            s_id = int(new_student_id.split('_')[0]) 
            dataset_index = int(new_student_id.split('_')[1]) 
            
            print(f"Processing new_student_id: {new_student_id}, s_id: {s_id}") 
            
            tool_name = tool_names[i]['select'] 
            student_test_data = self.get_student_test_data(new_student_id)
            history_questions, _ = select_question.build_answered_questions(student_test_data, s_id, self.id_maps, self.question_bank, self.test_results) 
            untested_questions = select_question.build_unanswered_questions(student_test_data, s_id, self.id_maps, self.question_bank) 

            if select_question.contains_string(untested_questions, tool_name): 
                tool_name = tool_name.replace('Pm_', '')
                s_question = int(tool_name)
                mapped_question = select_question.get_mapped_id(s_question, self.id_maps)
                
                pos_sid = s_id * 5 + dataset_index 
                tmp_pos_sid = (s_id % 16) * 5 + dataset_index
                replicated_student_id_to_accs_pos_map[new_student_id] = tmp_pos_sid 

                self.test_datas[pos_sid].apply_selection(0, mapped_question) 
                new_history_questions = select_question.build_answered_questions(student_test_data, s_id, self.id_maps, self.question_bank, self.test_results) 
                new_untested_questions = select_question.build_unanswered_questions(student_test_data, s_id, self.id_maps, self.question_bank) 
                
                formatted_history = []
                history_list, _ = new_history_questions
                for question in history_list:
                    content = question.split(" (")[0] 
                    is_correct = "incorrect" in question.lower()
                    formatted_question = {
                        "question": content,
                        "is_correct": not is_correct
                    }
                    formatted_history.append(formatted_question)
                
                student_tmp_report_request = {
                    "history_questions": formatted_history,
                    "student_id": new_student_id 
                }
                batch_reports_request.append(student_tmp_report_request)
                
                select_data = {
                    "ability_report": "Unable to generate ability report", 
                    "untested_questions": new_untested_questions,
                    "student_id": new_student_id 
                }
                batch_simulate_requests.append(select_data)

                results.append({
                    "history_questions": new_history_questions,
                    "student_id": new_student_id 
                })
                
                if dataset_index not in self.select_questions:
                    self.select_questions[dataset_index] = {}
                if s_id not in self.select_questions[dataset_index]: 
                    self.select_questions[dataset_index][s_id] = []
                self.select_questions[dataset_index][s_id].append(mapped_question)

            else: 
                results.append({
                    "history_questions": history_questions,
                    "student_id": new_student_id 
                })
        
        try:
            json_payload = json.dumps(batch_reports_request, ensure_ascii=False)
            response = requests.post(
                f"{self.llm_api_url}/batch_analyze_ability",
                data=json_payload,
                headers={
                    'Content-Type': 'application/json; charset=utf-8'
                }
            )
            batch_response_data = response.json()
            
            ability_report_map = {item["student_id"]: item.get("ability_report", "Unable to generate ability report") 
                                  for item in batch_response_data.get("reports", [])}
            
            for req_data in batch_simulate_requests:
                req_data["ability_report"] = ability_report_map.get(req_data["student_id"], "Unable to generate ability report")

        except Exception as e:
            print(f"Error calling LLM API for batch_analyze_ability: {e}")

        
        json_select_data = json.dumps(batch_simulate_requests, ensure_ascii=False)
        
        print("json select data",json_select_data)
        
        try:
            response = requests.post(
                f"{self.llm_api_url}/batch_simulate_answer",
                data=json_select_data,
                headers={
                    'Content-Type': 'application/json; charset=utf-8'
                }
            )
            
            batch_accuracies_results = response.json().get("f1_scores", []) 
            
            
            for accuracy_item in batch_accuracies_results:
                returned_student_id = accuracy_item["student_id"] 
                accuracy_value = accuracy_item["f1_score"] 

                if returned_student_id in replicated_student_id_to_accs_pos_map:
                    accs_pos_idx = replicated_student_id_to_accs_pos_map[returned_student_id]
                    accs[accs_pos_idx] = accuracy_value 

        except Exception as e:
            print(f"Error in simulate_answer request: {e}")
            


        return results, accs 

    def get_tool_final_results(self):
        return self.select_questions

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action (此方法在此上下文似乎未使用)
        """
        # valid tool call
        # print("selected questions: ", self.select_questions)
        if "results" in result:
            return 0.0
        else:
            return 0.0