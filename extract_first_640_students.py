import pandas as pd
import numpy as np
from pathlib import Path

def extract_first_640_students(input_file: str, output_file: str):
    """
    从CSV文件中提取前640个学生的数据并保存
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
    """
    try:
        print(f"正在读取CSV文件: {input_file}")
        
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        print(f"原始数据信息:")
        print(f"  总记录数: {len(df)}")
        print(f"  学生总数: {df['student_id'].nunique()}")
        print(f"  题目总数: {df['question_id'].nunique()}")
        
        # 获取所有唯一的学生ID
        unique_students = sorted(df['student_id'].unique())
        print(f"  唯一学生ID数量: {len(unique_students)}")
        
        # 提取前640个学生
        if len(unique_students) >= 640:
            selected_students = unique_students[:640]
        else:
            print(f"警告: 文件中只有 {len(unique_students)} 个学生，少于640个")
            selected_students = unique_students
        
        print(f"  选择的学生数量: {len(selected_students)}")
        
        # 筛选包含这些学生的所有记录
        filtered_df = df[df['student_id'].isin(selected_students)].copy()
        
        # 重新编号学生ID（从0开始）
        student_id_map = {old_id: new_id for new_id, old_id in enumerate(selected_students)}
        filtered_df['student_id'] = filtered_df['student_id'].map(student_id_map)
        
        # 保存到新的CSV文件
        filtered_df.to_csv(output_file, index=False)
        
        print(f"\n提取结果:")
        print(f"  提取的记录数: {len(filtered_df)}")
        print(f"  提取的学生数: {filtered_df['student_id'].nunique()}")
        print(f"  涉及的题目数: {filtered_df['question_id'].nunique()}")
        print(f"  数据已保存至: {output_file}")
        
        # 显示一些统计信息
        print(f"\n统计信息:")
        print(f"  平均每个学生的答题数: {len(filtered_df) / filtered_df['student_id'].nunique():.2f}")
        print(f"  正确答题比例: {filtered_df['correct'].mean():.4f}")
        
        # 显示前几个学生的答题情况
        print(f"\n前5个学生的答题情况:")
        for i in range(min(5, filtered_df['student_id'].nunique())):
            student_data = filtered_df[filtered_df['student_id'] == i]
            correct_count = student_data['correct'].sum()
            total_count = len(student_data)
            accuracy = correct_count / total_count if total_count > 0 else 0
            print(f"  学生 {i}: 答题 {total_count} 道，正确 {correct_count} 道，正确率 {accuracy:.4f}")
        
        return filtered_df
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None

def main():
    # 设置文件路径
    input_file = "datasets/MOOCRadar/train_triples.csv"
    output_file = "datasets/MOOCRadar/train_triples_0_639.csv"
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 输入文件 {input_file} 不存在")
        return
    
    # 执行提取
    result_df = extract_first_640_students(input_file, output_file)
    
    if result_df is not None:
        print(f"\n✅ 成功完成数据提取!")
    else:
        print(f"\n❌ 数据提取失败!")

if __name__ == "__main__":
    main() 