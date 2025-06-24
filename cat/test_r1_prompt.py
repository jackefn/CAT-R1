import CAT
import json
import torch
import numpy as np
import pandas as pd
import scipy.stats
from utils import select_question

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

# strategy = CAT.strategy.LLMTest()
ckpt_path = 'cat/ckpt/irt.pt'

# 加载数据
test_triplets = pd.read_csv(f'datasets/MOOCRadar/test_demo_triples.csv', encoding='utf-8')

# 按学生分组并将每个分组的学生ID设为0
student_dfs = []
for student_id, group in test_triplets.groupby('student_id'):
    student_df = group.copy()
    student_df['student_id'] = 0
    student_dfs.append(student_df)

test_datas = [] # 为每个学生创建AdapTestDataset
concept_map = json.load(open(f'datasets/MOOCRadar/concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'datasets/MOOCRadar/metadata_demo.json', 'r'))

id_maps_file = 'datasets/MOOCRadar/id_maps.json'
id_maps = select_question.load_id_maps(id_maps_file)

question_bank = json.load(open('datasets/MOOCRadar/question_bank.json', 'r'))

test_results = select_question.extract_question_correctness(student_dfs, id_maps) # 获取每个学生的所有答题结果
print(test_results)
# # 为每个分组创建AdapTestDataset
# for student_df in student_dfs:
#     student_data = student_df.to_records(index=False)
#     student_test_data = CAT.dataset.AdapTestDataset(
#         student_data, 
#         concept_map,
#         1,  
#         metadata['num_questions'], 
#         metadata['num_concepts']
#     )
#     test_datas.append(student_test_data)

# # 为每个学生创建指定的IRT模型
# models = []
# for test_data in test_datas:
#     model = CAT.model.IRTModel(**config)
#     model.init_model(test_data)
#     model.adaptest_load(ckpt_path)
#     models.append(model)
# i = 0 # 学生原始id（未归0）

# for test_data in test_datas: # 遍历每个学生的作答数据
#     final_result = ""
#     for it in range(10): # 设置最大迭代次数
#         batch_data = []
#         selected_questions = {} # 用于存储当前学生所选择的问题
#         str_id = select_question.get_original_id(i, id_maps)
#         str_id = 'U_' + str_id
#         model = models[i] # 加载指定学生对应的IRT模型
#         theta = model.get_theta(0)[0] # 获取指定学生的能力值
#         ability_percentile = scipy.stats.norm.cdf(theta) * 100 # 计算能力值的百分位数
#         if str_id in question_bank:
#             history_questions, _ = select_question.build_answered_questions(test_data, i, id_maps, question_bank, test_results) # 构造当前学生已选题库
#             untested_questions = select_question.build_unanswered_questions(test_data, i, id_maps, question_bank) # 构造当前学生未选题库
#             batch_data.append((ability_percentile, history_questions, untested_questions))
#         step_result = strategy.batch_get_chat(batch_data) # 初步进入单个学生的cat过程，输出可能是选题，也可能是最终的评估报告
#         selection = select_question.judge_selection(str(step_result)) # 判断是否处于选题动作
#         if selection is not None: # 目前处于选题动作
#             ex_selection = select_question.extract_response(selection)
#             if ex_selection is not None:
#                 if select_question.contains_string(batch_data[0][2], ex_selection): # 只有满足此条件才出发选题逻辑
#                     ex_response = ex_selection.replace('Pm_', '')
#                     s_question = int(ex_response)
#                     mapped_question = select_question.get_mapped_id(s_question, id_maps)
#                     selected_questions[0] = mapped_question
#                     print(selected_questions)
#                     for student, question in selected_questions.items():
#                         test_data.apply_selection(student, question) # 更新当前学生的作答数据
#                     models[i].adaptest_update(test_data) # 更新IRT模型
#                     final_result += str(step_result)
#                     final_result += "\n"
#                 else:
#                     final_result += str(step_result)
#                     final_result += "\n"
#                     final_result += "select uncorrect question"
#                     final_result += "\n"
#             else:
#                 final_result += "wrong format response"
#                 final_result += "\n"
#         else:
#             answer = select_question.judge_answer(str(step_result)) # 判断是否可以结束答题
#             if answer is not None:
#                 final_result += str(step_result)
#                 final_result += "\n"
#                 break
#     print(final_result)
#     i += 1
