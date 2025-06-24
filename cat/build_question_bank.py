import json
import re
import ast
def load_json_file(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            
            if file_path.endswith('problem.json'):
                data = {}
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data[item['problem_id']] = item
                    except json.JSONDecodeError:
                        continue
                return data
            
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None

def extract_content(data_str):
   
    match = re.search(r"'content':\s*'([^']*)'", data_str)
    if match:
        return match.group(1)  
    else:
        return None  

def process_student_problems(student_file, problem_data, result):
    
    student_data = load_json_file(student_file)
    if not student_data:
        return False
    
    
    for record in student_data:
        for problem in record['seq']:
            student_id = problem['user_id']
            problem_id = problem['problem_id']
            
            
            if student_id not in result:
                result[student_id] = []
            
            
            if problem_id in problem_data:
                problem_info = problem_data[problem_id]
                try:
                    
                    content = extract_content(problem_info['detail'])

                    if content:
                        content = content
                        print(content)
                    else:
                        content = "内容解析失败"
                except (KeyError, TypeError):
                    content = "内容解析失败"
                
                
                problem_dict = {
                    problem_id: {
                        "content": content,
                    }
                }
                
                
                if problem_dict not in result[student_id]:
                    result[student_id].append(problem_dict)
    
    return True

def main():
    
    student_files = [
        'datasets/MOOCRadar/student-problem-coarse.json',
        'datasets/MOOCRadar/student-problem-fine.json',
        'datasets/MOOCRadar/student-problem-middle.json'
    ]
    problem_file = 'datasets/MOOCRadar/problem.json'
    output_file = 'datasets/MOOCRadar/question_bank.json'  
    
    
    problem_data = load_json_file(problem_file)
    if not problem_data:
        print("加载题目数据失败")
        return
    
    
    result = {}
    
   
    for student_file in student_files:
        success = process_student_problems(student_file, problem_data, result)
        if not success:
            print(f"处理文件失败: {student_file}")
    
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"成功处理并保存文件: {output_file}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")

if __name__ == "__main__":
    main()