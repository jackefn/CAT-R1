import re
import json
def judge_selection(response): # 判断是否处于选题动作
    pattern = r'<select>(.*?)</select>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content
    else:
        return None

def judge_answer(response): 
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content
    else:
        return None

def contains_string(lst, target_string):
    if not isinstance(lst, list):
        raise TypeError("输入的变量必须是列表类型")
    for item in lst:
        if isinstance(item, dict) and target_string in item:
            return True
    return False
    
def extract_response(response):
        pattern = r'\bPm_\d+\b'
        match = re.search(pattern, response)
        if match:
            return match.group()
        else:
            return None

def load_id_maps(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        id_maps = json.load(f)
    return id_maps

def get_original_id(mapped_id, id_maps):
    student_map = id_maps.get('test_student_map', {})
    original_ids = student_map.get('original', [])
    mapped_ids = student_map.get('mapped', [])
    try:
        index = mapped_ids.index(mapped_id)
        return original_ids[index]
    except (ValueError, IndexError):
        return None

def get_original_question_id(mapped_id, id_maps):
    question_map = id_maps.get('question_map', {})
    original_ids = question_map.get('original', [])
    mapped_ids = question_map.get('mapped', [])
    try:
        index = mapped_ids.index(mapped_id)
        return original_ids[index]
    except (ValueError, IndexError):
        return None

def extract_is_correct(student_dfs, id_maps):
    results = {}
    for i, student_df in enumerate(student_dfs):
        original_id = get_original_id(i, id_maps)
        if original_id is None:
            continue
        records = student_df.to_dict('records')
        results[original_id] = [int(record['correct']) for record in records]
    
    return results

def get_mapped_id(original_id, id_maps):
    question_map = id_maps.get('question_map', {})
    original_ids = question_map.get('original', [])
    mapped_ids = question_map.get('mapped', [])
    try:
        original_id = str(original_id)
        index = original_ids.index(original_id)
        return mapped_ids[index]
    except (ValueError, IndexError):
        return None

def build_answered_questions(test_data, student_id, id_maps, question_bank, test_results):
    answered_questions = []
    question_contents = []
    
    str_id = str(get_original_id(student_id, id_maps))  # 获取原始学生id
    tested_data = test_data.tested[0]  # 获取已选题目id（数字形式）
    student_results = test_results.get(str_id, {})  # 获取学生答题结果
    
    # 构建题目内容和答题记录
    for q_id in tested_data:
        original_q_id = 'Pm_' + str(get_original_question_id(q_id, id_maps)) # 获取原始已选问题id
        
        # 在题库中查找题目内容
        for question_group in question_bank.get('U_'+str_id, []):  # 注意这里是遍历question_group
            if original_q_id in question_group:
                question_data = question_group[original_q_id]
                question_contents.append({
                    'id': original_q_id,
                    'content': question_data['content']
                })
                break
    
    # 构建历史答题记录
    history = []
    for q_content in question_contents:
        q_id = q_content['id'].replace('Pm_', '')  # 移除Pm_前缀
        q_id = get_mapped_id(q_id,id_maps)
        correction = student_results.get(str(q_id), 0)  # 获取答题结果
        status = '(correct)' if correction == 1 else '(incorrect)'
        history.append(f"{q_content['content']} {status}")
    
    return history, question_contents

def build_unanswered_questions(test_data, student_id, id_maps, question_bank):
    unanswered = []
    str_id = 'U_' + str(get_original_id(student_id, id_maps))
    untested_ids = test_data.untested[0]
    for q_id in untested_ids:
        original_q_id = 'Pm_' + str(get_original_question_id(q_id, id_maps))
        for question in question_bank.get(str_id, []):
            if original_q_id in question:
                unanswered.append(question)
                break
    return unanswered

def extract_question_correctness(student_dfs, id_maps):
    results = {}
    for i, student_df in enumerate(student_dfs):
        original_id = get_original_id(i, id_maps)
        if original_id is None:
            continue
        if 'question_id' not in student_df.columns or 'correct' not in student_df.columns:
            raise ValueError("DataFrame 必须包含 'question_id' 和 'correct' 列")
        question_correctness = {
            str(record['question_id']): int(record['correct'])
            for record in student_df.to_dict('records')
        }
        results[original_id] = question_correctness
    return results