import cat.CAT as CAT
import json
import torch
import numpy as np
import pandas as pd
import scipy.stats
import select_question
from agent.tool.tools.llm_select_strategy import LLMSelectStrategy

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

test_length = 10
strategy = LLMSelectStrategy()
ckpt_path = 'cat/ckpt/irt.pt'

test_triplets = pd.read_csv(f'datasets/MOOCRadar/test_triples.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'datasets/MOOCRadar/concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'datasets/MOOCRadar/metadata.json', 'r'))
test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                        metadata['num_test_students'], 
                                        metadata['num_questions'], 
                                        metadata['num_concepts'])

import warnings
warnings.filterwarnings("ignore")

model = CAT.model.IRTModel(**config)
model.init_model(test_data)
model.adaptest_load(ckpt_path)

# 加载必要的映射和题库数据
id_maps_file = 'datasets/MOOCRadar/id_maps.json'
id_maps = json.load(open(id_maps_file, 'r'))
question_bank = json.load(open('datasets/MOOCRadar/question_bank.json', 'r'))

# 获取所有学生的答题结果
student_dfs = []
for student_id, group in pd.DataFrame(test_triplets).groupby('student_id'):
    student_df = group.copy()
    student_df['student_id'] = 0
    student_dfs.append(student_df)
test_results = select_question.extract_question_correctness(student_dfs, id_maps)

S_sel = {}
for sid in range(test_data.num_students): 
    key = sid
    S_sel[key] = []

for it in range(1, test_length + 1):
    print(f"Iteration {it}")
    # 获取所有需要测试的学生ID
    student_ids = [str(sid) for sid in range(test_data.num_students)]
    
    # 使用LLM策略批量选择题目
    selected_questions = strategy.batch_select_questions(
        test_data, 
        student_ids, 
        id_maps, 
        question_bank, 
        test_results
    )
    
    # 应用选择的题目
    for student, question in selected_questions.items():
        test_data.apply_selection(student, question)
        S_sel[student].append(question)
    
    # 更新模型
    model.adaptest_update(test_data)
    
    # 评估结果
    results = model.evaluate(test_data)
    print(results)