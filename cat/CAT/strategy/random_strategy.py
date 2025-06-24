import numpy as np
import logging

from cat.CAT.strategy.abstract_strategy import AbstractStrategy
from cat.CAT.model import AbstractModel
from cat.CAT.dataset import AdapTestDataset


class RandomStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Random Select Strategy'

    def adaptest_select(self, adaptest_data: AdapTestDataset):
        selection = {}
        students_without_questions = []  # 记录没有可选题目的学生ID
        
        for sid in range(adaptest_data.num_students):
            # 获取未测试的题目
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            
            # 检查是否还有未测试的题目
            if len(untested_questions) == 0:
                students_without_questions.append(sid)  # 记录该学生ID
                # logging.info(f"学生{sid}没有可选择的题目")
                # logging.info(f"该学生已完成的题目数量: {len(adaptest_data.tested[sid])}")
                continue
            
            # 随机选择一道题目
            random_idx = np.random.randint(len(untested_questions))
            selection[sid] = untested_questions[random_idx]
        
        # 如果有学生没有可选题目，输出详细信息
        # if students_without_questions:
        #     logging.warning(f"以下学生没有可选择的题目：{students_without_questions}")
        #     logging.warning(f"总学生数：{adaptest_data.num_students}")
        #     logging.warning(f"还有题目可选的学生数：{len(selection)}")
        
        # 检查是否所有学生都完成了测试
        if not selection:
            raise ValueError(f"所有{adaptest_data.num_students}个学生都已完成全部题目，测试结束")
        
        return selection