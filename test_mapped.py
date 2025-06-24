from agent.tool.tools import select_question
id_maps_file = 'datasets/MOOCRadar/id_maps.json'
id_maps = select_question.load_id_maps(id_maps_file) # 加载id映射文件
question_id=3243760
original_question_id = select_question.get_mapped_id(int(question_id), id_maps)
print("original_question_id: ", original_question_id)