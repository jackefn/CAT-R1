import json
import os
import pandas as pd
import re
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
import uvicorn
from vllm import LLM, SamplingParams
import select_question
from select_question import judge_selection, judge_answer, contains_string, extract_response, load_id_maps, get_original_id, get_original_question_id, build_answered_questions, build_unanswered_questions, extract_question_correctness, get_original_student_id, get_original_question_id_new, get_mapped_student_id, get_mapped_question_id, build_answered_questions_new, build_unanswered_questions_new
import asyncio
import aiohttp
from tqdm import tqdm

app = FastAPI(title="Qwen-2.5-1.5B for Student Modeling")
id_maps_file = 'datasets/MOOCRadar/id_maps_balanced.json'
id_maps = select_question.load_id_maps(id_maps_file) # 加载id映射文件
test_data_file = 'datasets/MOOCRadar/test_triples_balanced.csv'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

llm = LLM(
    model="/d2/mxy/Models/Qwen2-7B",
    tokenizer="/d2/mxy/Models/Qwen2-7B",
    tensor_parallel_size=1,
    max_model_len=32000,
    enforce_eager=True,
    gpu_memory_utilization=0.6
)

base_params = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    max_tokens=32000,
    skip_special_tokens=True
)

def build_chat_template(messages: List[dict]) -> str:
    template = ""
    for msg in messages:
        if msg["role"] == "system":
            template += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            template += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
    template += "<|im_start|>assistant\n"
    return template


class AbilityRequest(BaseModel):
    history_questions: List[Dict]  
    student_id: str

class BatchAbilityRequests(RootModel[List[AbilityRequest]]):
   pass


class AnswerRequest(BaseModel):
    ability_report: str         
    untested_questions: List[Dict[str, Dict[str, str]]]
    student_id: str             

class BatchAnswerRequests(RootModel[List[AnswerRequest]]):
    pass


class StudentHistoryData(BaseModel):
    student_id: str
    history_questions: List[str]  
    untested_questions: List[Dict]  

class BatchStudentData(BaseModel):
    students: List[StudentHistoryData]

class StudentAnswerData(BaseModel):
    student_id: str
    ability_report: str
    untested_questions: List[Dict]

class BatchAnswerData(BaseModel):
    students: List[StudentAnswerData]

try:
    GLOBAL_TEST_DATA = pd.read_csv(test_data_file)
    print(f"Successfully loaded test data from {test_data_file}")
except FileNotFoundError:
    print(f"Error: {test_data_file} not found. Please check the path.")
    GLOBAL_TEST_DATA = pd.DataFrame() 
except Exception as e:
    print(f"Error loading test data: {e}")
    GLOBAL_TEST_DATA = pd.DataFrame()

def get_correct_correctness(original_student_id, original_question_id):
    if GLOBAL_TEST_DATA.empty:
        print("Warning: GLOBAL_TEST_DATA is empty. Cannot get ground truth correctness.")
        return None
    
    result = GLOBAL_TEST_DATA[(GLOBAL_TEST_DATA['student_id'] == int(original_student_id)) & 
                              (GLOBAL_TEST_DATA['question_id'] == int(original_question_id))]
    if not result.empty:
        return result['correct'].iloc[0]
    return None

def build_ability_report_prompt(history: List[Dict], student_id: str) -> str:
    """
    根据学生的历史记录和ID构建用于能力报告的单个提示词。
    """
    prompt = f"""Based on the following student's (ID: {student_id}) answer history, analyze their current ability level and generate a detailed report.

                    Answer History:
                    {json.dumps(history, indent=2, ensure_ascii=False)}

                    Please analyze:
                    1. Overall performance level (percentile)
                    2. Strengths and weaknesses
                    3. Knowledge mastery in different areas
                    4. Learning progress and trends

                    Generate a comprehensive report:"""

    messages = [
        {"role": "system", "content": "You are an expert educational analyst."},
        {"role": "user", "content": prompt}
    ]
    return build_chat_template(messages)


def generate_ability_report(history: List[Dict], student_id: str) -> str:
    
    prompt = f"""Based on the following student's (ID: {student_id}) answer history, analyze their current ability level and generate a detailed report. 
    
                    Answer History:
                    {json.dumps(history, indent=2,ensure_ascii=False)}

                    Please analyze:
                    1. Overall performance level (percentile)
                    2. Strengths and weaknesses
                    3. Knowledge mastery in different areas
                    4. Learning progress and trends

                    Generate a comprehensive report:"""

    messages = [
        {"role": "system", "content": "You are an expert educational analyst."},
        {"role": "user", "content": prompt}
    ]
    input_text = build_chat_template(messages)
    
    # 使用vLLM生成
    outputs = llm.generate([input_text], base_params)
    response = outputs[0].outputs[0].text.strip()
    
    return response.split("<|im_end|>")[0].strip()

@app.post("/batch_analyze_ability")
async def batch_analyze_ability_endpoint(batch_req: BatchAbilityRequests):
    reports = []
    prompts_to_generate = []
    # 用于在收到 LLM 批量输出后，将报告与原始 student_id 关联起来
    student_id_map = []

    # 1. 遍历每个学生的请求，构建各自的 prompt，并收集到列表中
    request_list = batch_req.root
    for req in request_list: 
        prompt = build_ability_report_prompt(req.history_questions, req.student_id)
        prompts_to_generate.append(prompt)
        student_id_map.append(req.student_id) # 记录对应的学生ID

    try:
        # 2. 使用 vLLM 进行批量生成
        # llm.generate 接收一个 prompt 字符串的列表
        outputs = llm.generate(prompts_to_generate, base_params)
        # 3. 解析并汇总 LLM 的批量输出
        # outputs 是一个列表，每个元素对应 prompts_to_generate 中的一个 prompt
        for i, output in enumerate(outputs):
            # 提取 LLM 的文本输出，并清除特殊标记
            report_text = output.outputs[0].text.strip().split("<|im_end|>")[0].strip()
            reports.append({
                "student_id": student_id_map[i], # 使用之前保存的 student_id 映射结果
                "ability_report": report_text
            })
        # 4. 返回批量响应
        return {"reports": reports}

    except Exception as e:
        # 批量请求失败时的错误处理
        raise HTTPException(status_code=500, detail=f"Error during batch ability analysis: {str(e)}")

@app.post("/analyze_ability")
async def analyze_ability(request: AbilityRequest):
    try:
        report = generate_ability_report(request.history_questions, request.student_id)
        return {"ability_report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_analyze_ability_v2")
async def batch_analyze_ability_v2_endpoint(batch_req: BatchStudentData):

    reports = []
    prompts_to_generate = []
    student_id_map = []

    # 处理每个学生的请求
    for student_data in batch_req.students:
        # 将历史题目格式化为API需要的格式
        formatted_history = []
        for question in student_data.history_questions:
            # 解析问题内容和答题结果
            if " (correct)" in question:
                content = question.replace(" (correct)", "")
                is_correct = True
            elif " (incorrect)" in question:
                content = question.replace(" (incorrect)", "")
                is_correct = False
            else:
                content = question
                is_correct = False  # 默认值
            
            formatted_history.append({
                "question": content,
                "is_correct": is_correct
            })
        
        # 构建提示词
        prompt = build_ability_report_prompt(formatted_history, student_data.student_id)
        prompts_to_generate.append(prompt)
        student_id_map.append(student_data.student_id)

    try:
        # 使用vLLM进行批量生成
        outputs = llm.generate(prompts_to_generate, base_params)
        # 处理输出结果
        for i, output in enumerate(outputs):
            report_text = output.outputs[0].text.strip().split("<|im_end|>")[0].strip()
            reports.append({
                "student_id": student_id_map[i],
                "ability_report": report_text
            })
        
        return {"reports": reports}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch ability analysis: {str(e)}")

@app.post("/batch_simulate_answer_v2")
async def batch_simulate_answer_v2_endpoint(batch_req: BatchAnswerData):
    """
    适配版本的批量答题预测接口
    接收格式: {"students": [{"student_id": "123", "ability_report": "...", "untested_questions": [...]}]}
    返回每个学生的F1分数
    """
    results = []
    all_prompts_info = []
    student_question_ranges = []
    
    # 处理每个学生的请求
    for student_req_idx, student_data in enumerate(batch_req.students):
        current_student_prompts_start_idx = len(all_prompts_info)
        
        # 遍历当前学生的未作答问题
        for question_dict in student_data.untested_questions:
            # 处理题目字典格式
            if isinstance(question_dict, dict):
                for key, value in question_dict.items():
                    if isinstance(value, dict) and "content" in value:
                        question_id = key.replace("Pm_", "")
                        question_content = value["content"]
                        
                        # 构建预测提示词
                        prompt = f"""Based on the student's ability assessment report and the given question, analyze whether the student can correctly answer this question.

Student's Ability Report:
{student_data.ability_report}

Question Content:
{question_content}

Please analyze:
1. Whether the knowledge points in this question match the student's mastered areas
2. Whether the question's difficulty level aligns with the student's current ability
3. Consider the student's performance pattern in similar topics

Based on the above analysis, predict:
Will the student answer this question correctly?

You should only answer with boxed(Yes) or boxed(No), don't add any other words."""
                        
                        messages = [
                            {"role": "system", "content": "You are a student modeler."},
                            {"role": "user", "content": prompt}
                        ]
                        
                        all_prompts_info.append({
                            "student_id": student_data.student_id,
                            "question_id_raw": question_id,
                            "prompt_text": build_chat_template(messages)
                        })
        
        student_question_ranges.append({
            "student_id": student_data.student_id,
            "start_idx": current_student_prompts_start_idx,
            "end_idx": len(all_prompts_info)
        })
    
    if not all_prompts_info:
        return {"f1_scores": []}
    
    try:
        prompts_for_llm = [item["prompt_text"] for item in all_prompts_info]
        outputs = llm.generate(prompts_for_llm, base_params)
        
        # 解析预测结果
        all_predictions = []
        pattern = r'(?:boxed\((.*?)\)|\b(yes|no)\b)'
        
        for output in outputs:
            response = output.outputs[0].text.strip()
            print("simulate answer response: ", response)
            predict_correctness = None
            match = re.search(pattern, response, re.IGNORECASE)
            
            if match:
                extracted_text = match.group(1) if match.group(1) else match.group(2)
                if extracted_text:
                    prediction = extracted_text.lower()
                    predict_correctness = 1 if prediction == "yes" else 0
            all_predictions.append(predict_correctness)
        
        # 计算每个学生的F1分数
        for student_range in student_question_ranges:
            student_id = student_range["student_id"]
            start_idx = student_range["start_idx"]
            end_idx = student_range["end_idx"]
            
            # 初始化混淆矩阵的各个值
            tp = 0  # True Positive: 预测正确且实际正确
            fp = 0  # False Positive: 预测正确但实际错误
            fn = 0  # False Negative: 预测错误但实际正确
            tn = 0  # True Negative: 预测错误且实际错误
            
            for i in range(start_idx, end_idx):
                current_prompt_info = all_prompts_info[i]
                llm_predicted_correctness = all_predictions[i]
                original_id_maps = select_question.load_id_maps("datasets/MOOCRadar/id_map.json")
                question_id_tmp = select_question.get_mapped_id(int(current_prompt_info["question_id_raw"]), original_id_maps)
                original_question_id = select_question.get_mapped_question_id(int(question_id_tmp), id_maps)
                ground_truth_correctness = get_correct_correctness(int(student_id), original_question_id)
                print("ground_truth_correctness: ", ground_truth_correctness)
                if llm_predicted_correctness is not None and ground_truth_correctness is not None:
                    # 计算混淆矩阵
                    if llm_predicted_correctness == 1 and ground_truth_correctness == 1:
                        tp += 1
                    elif llm_predicted_correctness == 1 and ground_truth_correctness == 0:
                        fp += 1
                    elif llm_predicted_correctness == 0 and ground_truth_correctness == 1:
                        fn += 1
                    elif llm_predicted_correctness == 0 and ground_truth_correctness == 0:
                        tn += 1
            
            # 计算精确率、召回率和F1分数
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print("F1 score: ", f1_score)
            results.append({
                "student_id": student_id,
                "f1_score": f1_score
            })
        
        return {"f1_scores": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during batch answer prediction: {str(e)}")

@app.post("/batch_simulate_answer")
async def batch_simulate_answer_endpoint(batch_req: BatchAnswerRequests):
    """
    接收一批学生的答题数据，批量推理并计算每个学生的 LLM 评估F1分数。
    """
    
    results = []
    # 存储所有需要 LLM 推理的 prompts，以及它们对应的学生请求索引和问题索引
    # 格式：[(student_req_idx, question_idx, prompt_text), ...]
    all_prompts_info = [] 
    
    # 用于记录每个学生有多少个未答问题，以及它们在 all_prompts_info 中的起始/结束索引
    student_question_ranges = [] 
    
    request_list = batch_req.root
    for student_req_idx, req in enumerate(request_list):
        current_student_prompts_start_idx = len(all_prompts_info)
        
        # 遍历当前学生的未作答问题，构建 LLM 预测的 prompts
        for question_dict in req.untested_questions:
            for key, value in question_dict.items():
                question_id = key.replace("Pm_", "")
                question_content = value["content"]

                # 确保 ability_report 是字符串格式，如果它是 Dict，需要序列化
                ability_report_str = json.dumps(req.ability_report, indent=2, ensure_ascii=False)

                prompt = f"""Based on the student's ability assessment report and the given question, analyze whether the student can correctly answer this question.

                    Student's Ability Report:
                    {ability_report_str}

                    Question Content:
                    {question_content}

                    Please analyze:
                    1. Whether the knowledge points in this question match the student's mastered areas
                    2. Whether the question's difficulty level aligns with the student's current ability
                    3. Consider the student's performance pattern in similar topics

                    Based on the above analysis, predict:
                    Will the student answer this question correctly?

                    You should only answer with boxed(Yes) or boxed(No), don't add any other words.
                    """
                messages = [
                    {"role": "system", "content": "You are a student modeler."},
                    {"role": "user", "content": prompt}
                ]
                all_prompts_info.append({
                    "student_id": req.student_id,
                    "question_id_raw": question_id, 
                    "prompt_text": build_chat_template(messages)
                })
        student_question_ranges.append({
            "student_id": req.student_id,
            "start_idx": current_student_prompts_start_idx,
            "end_idx": len(all_prompts_info) 
        })
    if not all_prompts_info:
        return {"f1_scores": []} 

    prompts_for_llm = [item["prompt_text"] for item in all_prompts_info]
    
    try:
        outputs = llm.generate(prompts_for_llm, base_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during LLM batch generation: {str(e)}")

    all_predictions = []
    pattern = r'(?:boxed\((.*?)\)|\b(yes|no)\b)'

    for output in outputs:
        response = output.outputs[0].text.strip()
        predict_correctness = None
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            extracted_text = match.group(1) if match.group(1) else match.group(2)
            if extracted_text:
                prediction = extracted_text.lower()
                predict_correctness = 1 if prediction == "yes" else 0
        all_predictions.append(predict_correctness)

    # 计算每个学生的F1分数
    for student_range in student_question_ranges:
        student_id = student_range["student_id"]
        start_idx = student_range["start_idx"]
        end_idx = student_range["end_idx"]
        
        # 初始化混淆矩阵的各个值
        tp = 0  # True Positive: 预测正确且实际正确
        fp = 0  # False Positive: 预测正确但实际错误
        fn = 0  # False Negative: 预测错误但实际正确
        tn = 0  # True Negative: 预测错误且实际错误
        
        for i in range(start_idx, end_idx):
            current_prompt_info = all_prompts_info[i]
            llm_predicted_correctness = all_predictions[i]
            
            original_question_id = select_question.get_mapped_id(int(current_prompt_info["question_id_raw"]), id_maps)
            ground_truth_correctness = get_correct_correctness(int(student_id), original_question_id)
            
            if llm_predicted_correctness is not None and ground_truth_correctness is not None:
                # 计算混淆矩阵
                if llm_predicted_correctness == 1 and ground_truth_correctness == 1:
                    tp += 1
                elif llm_predicted_correctness == 1 and ground_truth_correctness == 0:
                    fp += 1
                elif llm_predicted_correctness == 0 and ground_truth_correctness == 1:
                    fn += 1
                elif llm_predicted_correctness == 0 and ground_truth_correctness == 0:
                    tn += 1
        
        # 计算精确率、召回率和F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print("F1 score: ", f1_score)
        results.append({
            "student_id": student_id,
            "f1_score": f1_score
        })
        
    
    return {"f1_scores": results}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8101)
