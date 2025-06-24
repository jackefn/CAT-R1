import json
import torch
import numpy as np
import pandas as pd
import requests
import CAT
from typing import Dict, List
import select_question

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


API_BASE_URL = "http://localhost:8005"
concept_map = json.load(open(f'datasets/MOOCRadar/concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'datasets/MOOCRadar/metadata_test.json', 'r'))

def get_ability_report(history_questions: List[Dict], student_id: str) -> str:
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze_ability",
            json={"history_questions": history_questions, "student_id": student_id}
        )
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"Error getting ability report: {e}")
        return None

def simulate_answer(history_questions: List, ability_report: str, 
                   untested_questions: List[Dict], student_id: str) -> float:
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/simulate_answer",
            json={
                "history_questions": history_questions,
                "ability_report": ability_report,
                "untested_questions": untested_questions,
                "student_id": student_id
            }
        )
        response.raise_for_status()
        return response.json()["accuracy"]
    except Exception as e:
        print(f"Error simulating answers: {e}")
        return None

def format_history_for_api(history_questions: List[str]) -> List[Dict]:
    formatted_history = []
    for question in history_questions:
        content = question.split(" (")[0]
        is_correct = "(correct)" in question
        
        formatted_question = {
            "question": content,
            "is_correct": is_correct
        }
        formatted_history.append(formatted_question)
    return formatted_history

test_triplets = pd.read_csv('datasets/MOOCRadar/test_triples.csv', encoding='utf-8')

def format_history_for_simulate(history_questions: List[str]) -> List[List[str]]:
    formatted_history = []
    for question in history_questions:
        # 分割题目内容和状态
        content = question.split(" (")[0]
        status = "(correct)" if "(correct)" in question else "(incorrect)"
        # 将每个题目转换为包含两个元素的列表
        formatted_history.append([content, status])
    return formatted_history


test_datas = []
student_dfs = []
for student_id, group in test_triplets.groupby('student_id'):
    student_df = group.copy()
    student_df['student_id'] = 0
    student_dfs.append(student_df)
    
    student_data = student_df.to_records(index=False)
    student_test_data = CAT.dataset.AdapTestDataset(
        student_data, 
        concept_map,
        1,  
        metadata['num_questions'], 
        metadata['num_concepts']
    )
    test_datas.append(student_test_data)

id_maps_file = 'datasets/MOOCRadar/id_maps.json'
id_maps = select_question.load_id_maps(id_maps_file)
test_results = select_question.extract_question_correctness(student_dfs, id_maps)
            
question_bank = json.load(open('datasets/MOOCRadar/question_bank.json', 'r'))
test_length = 10
strategy = CAT.strategy.RandomStrategy() # 传统选题策略


student_history = {}  
accuracies = []      

for t in range(0, 1823): 
    student_history[t] = []

for i in range(test_length):
    print(f"\nIteration {i+1}")
    accs = []
    sum_acc = 0
    for s_id, test_data in enumerate(test_datas):
        selected_questions = strategy.adaptest_select(test_data)
        print(f"Student {s_id} selected questions:", selected_questions)
        
        for student, question in selected_questions.items():
            test_data.apply_selection(student, question)
            
        history_questions, _ = select_question.build_answered_questions(
            test_data, s_id, 
            id_maps, 
            question_bank, 
            test_results
        )
        formatted_history = format_history_for_api(history_questions)
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze_ability",
                json={
                    "history_questions": formatted_history,
                    "student_id": str(s_id)
                },
                headers={
                    'Content-Type': 'application/json; charset=utf-8'
                }
            )
            response.raise_for_status()
            ability_report = response.json()["ability_report"]
            # print(f"\nStudent {s_id} Ability Report:")
            # print(ability_report)
            untested_questions = select_question.build_unanswered_questions(
                test_data, s_id, id_maps, question_bank
            )
            simulate_response = requests.post(
                f"{API_BASE_URL}/simulate_answer",
                json={
                    "history_questions": format_history_for_simulate(history_questions),
                    "ability_report": ability_report,
                    "untested_questions": untested_questions,
                    "student_id": str(s_id)
                },
                headers={
                    'Content-Type': 'application/json; charset=utf-8'
                }
            )
            accuracy = simulate_response.json()["accuracy"]
            print(f"Student {s_id} accuracy: {accuracy}")
            accs.append(accuracy)
    
        except Exception as e:
            print(f"Error calling API for student {s_id}: {e}")
    for a in accs:
        if a is not None:
            sum_acc += a
    avg_acc = sum_acc / len(accs)
    print(f"Average accuracy: {avg_acc}")