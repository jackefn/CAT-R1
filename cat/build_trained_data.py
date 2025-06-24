import CAT
import json
import torch
import numpy as np
import pandas as pd
import scipy.stats
from utils import select_question
import os

prefix_prompt = """
    Your task is to simulate the entire process of a computer-adaptive test. 
    You can select questions for the student based on their current answers, the cognitive diagnostic model’s assessment of the student’s ability, and the questions they have already selected. 
    You can also decide whether the student still needs further assessment.
    You must first reason within <think>...</think>. 
    If you believe the student still needs more questions to be assessed more accurately, you can output a selected question for the student within <select>...</select> after the <think>...</think> part.
    Remember, you can only select one question at a time.
    When you believe that the student no longer requires assessment, you can output the final evaluation report for the student within <answer>...</answer>.
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
    student's ID: {student_id}
    student's ability: {theta}
    history questions: {history}
    candidate questions: {untested}
"""
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
dataset = 'moocradar'
config = {
    'learning_rate': 0.0025,
    'batch_size': 2048,
    'num_epochs': 8,
    'num_dim': 1, 
    'device': 'cuda:0',
    'policy':'',
    'betas': (0.9, 0.999),
}

ckpt_path = 'cat/ckpt/irt.pt'

test_triplets = pd.read_csv(f'datasets/MOOCRadar/test_demo_triples.csv', encoding='utf-8')

student_dfs = []
for student_id, group in test_triplets.groupby('student_id'):
    student_df = group.copy()
    student_df['student_id'] = 0
    student_dfs.append(student_df)

concept_map = json.load(open(f'datasets/MOOCRadar/concept_map.json', 'r'))
concept_map = {int(k): v for k, v in concept_map.items()}
metadata = json.load(open(f'datasets/MOOCRadar/metadata_demo.json', 'r'))
id_maps_file = 'datasets/MOOCRadar/id_maps.json'
id_maps = select_question.load_id_maps(id_maps_file)
question_bank = json.load(open('datasets/MOOCRadar/question_bank.json', 'r'))

test_datas = []
for student_df in student_dfs:
    student_data = student_df.to_records(index=False)
    student_test_data = CAT.dataset.AdapTestDataset(
        student_data, 
        concept_map,
        1,  
        metadata['num_questions'], 
        metadata['num_concepts']
    )
    test_datas.append(student_test_data)

models = []
for test_data in test_datas:
    model = CAT.model.IRTModel(**config)
    model.init_model(test_data)
    model.adaptest_load(ckpt_path)
    models.append(model)

prompts = []

i = 0  # 学生原始id（未归0）
for test_data in test_datas:
    str_id = select_question.get_original_id(i, id_maps)
    str_id = 'U_' + str_id
    model = models[i]  # 加载指定学生对应的IRT模型
    theta = model.get_theta(0)[0]  # 获取指定学生的能力值
    ability_percentile = scipy.stats.norm.cdf(theta) * 100  # 计算能力值的百分位数
    if str_id in question_bank:
        untested_questions = select_question.build_unanswered_questions(test_data, i, id_maps, question_bank)  # 构造当前学生未选题库
        prompt_content = prefix_prompt.format(student_id=i, theta=ability_percentile, untested=untested_questions, history=[])
        prompts.append({
            "data_source": dataset,  # 添加 data_source 字段
            "prompt": [  # 将 prompt 包装为列表
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        })
    i += 1
    
output_file = f'datasets/MOOCRadar/trained_data_demo.parquet'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
prompts_df = pd.DataFrame(prompts)
prompts_df.to_parquet(output_file, index=False)