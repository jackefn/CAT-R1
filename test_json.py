import json

# 创建包含中文的字典
chinese_data = {
    "name": "张三",
    "questions": [
        {
            "id": "Q1",
            "content": "什么是机器学习?",
            "answer": "机器学习是人工智能的一个分支",
            "is_correct": True
        },
        {
            "id": "Q2", 
            "content": "深度学习是什么?",
            "answer": "深度学习是机器学习的一个子领域",
            "is_correct": False
        }
    ],
    "subjects": {
        "math": "数学",
        "physics": "物理",
        "chemistry": "化学"
    },
    "description": "这是一个测试数据"
}


json_str = json.dumps(chinese_data, ensure_ascii=False, indent=2)
print("JSON字符串:")
print(json_str)