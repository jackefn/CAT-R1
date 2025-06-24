import pandas as pd
import numpy as np
from pathlib import Path

def calculate_answer_probability(csv_file_path: str, output_file_path: str):
    """
    计算CSV文件中每个学生的答题概率和整体平均率
    
    Args:
        csv_file_path: CSV文件路径
        output_file_path: 输出txt文件路径
    """
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        # 检查必要的列是否存在
        required_columns = ['student_id', 'question_id', 'correct']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
        
        # 按学生ID分组计算答题统计
        student_stats = df.groupby('student_id').agg({
            'correct': ['count', 'sum']
        }).round(4)
        
        # 重命名列
        student_stats.columns = ['total_questions', 'correct_answers']
        student_stats['wrong_answers'] = student_stats['total_questions'] - student_stats['correct_answers']
        
        # 计算概率
        student_stats['correct_probability'] = (student_stats['correct_answers'] / student_stats['total_questions']).round(4)
        student_stats['wrong_probability'] = (student_stats['wrong_answers'] / student_stats['total_questions']).round(4)
        
        # 计算整体平均率
        total_questions = df['correct'].count()
        total_correct = df['correct'].sum()
        total_wrong = total_questions - total_correct
        
        overall_correct_rate = (total_correct / total_questions).round(4)
        overall_wrong_rate = (total_wrong / total_questions).round(4)
        
        # 写入结果到txt文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("学生答题概率统计报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 写入每个学生的统计信息
            f.write("各学生答题统计:\n")
            f.write("-" * 30 + "\n")
            for student_id, stats in student_stats.iterrows():
                f.write(f"学生ID: {student_id}\n")
                f.write(f"  总题目数: {stats['total_questions']}\n")
                f.write(f"  答对题目数: {stats['correct_answers']}\n")
                f.write(f"  答错题目数: {stats['wrong_answers']}\n")
                f.write(f"  答对概率: {stats['correct_probability']:.4f} ({stats['correct_probability']*100:.2f}%)\n")
                f.write(f"  答错概率: {stats['wrong_probability']:.4f} ({stats['wrong_probability']*100:.2f}%)\n")
                f.write("\n")
            
            # 写入整体统计信息
            f.write("整体统计信息:\n")
            f.write("-" * 30 + "\n")
            f.write(f"总题目数: {total_questions}\n")
            f.write(f"总答对题目数: {total_correct}\n")
            f.write(f"总答错题目数: {total_wrong}\n")
            f.write(f"平均答对率: {overall_correct_rate:.4f} ({overall_correct_rate*100:.2f}%)\n")
            f.write(f"平均答错率: {overall_wrong_rate:.4f} ({overall_wrong_rate*100:.2f}%)\n")
            
            # 写入学生数量统计
            f.write(f"\n学生总数: {len(student_stats)}\n")
            
            # 写入答题情况分布
            f.write("\n答题情况分布:\n")
            f.write("-" * 30 + "\n")
            correct_distribution = student_stats['correct_probability'].value_counts().sort_index()
            f.write("答对概率分布:\n")
            for prob, count in correct_distribution.items():
                f.write(f"  答对概率 {prob:.2f}: {count} 名学生\n")
        
        print(f"统计结果已保存到: {output_file_path}")
        
        # 在控制台输出整体平均率
        print("\n" + "=" * 50)
        print("整体统计结果:")
        print(f"总题目数: {total_questions}")
        print(f"总答对题目数: {total_correct}")
        print(f"总答错题目数: {total_wrong}")
        print(f"平均答对率: {overall_correct_rate:.4f} ({overall_correct_rate*100:.2f}%)")
        print(f"平均答错率: {overall_wrong_rate:.4f} ({overall_wrong_rate*100:.2f}%)")
        print(f"学生总数: {len(student_stats)}")
        print("=" * 50)
        
        return {
            'total_questions': total_questions,
            'total_correct': total_correct,
            'total_wrong': total_wrong,
            'overall_correct_rate': overall_correct_rate,
            'overall_wrong_rate': overall_wrong_rate,
            'student_count': len(student_stats)
        }
        
    except FileNotFoundError:
        print(f"错误: 找不到CSV文件 {csv_file_path}")
        return None
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None

def main():
    # 设置文件路径
    csv_file_path = "/d2/mxy/cat-r1/cat-r1/datasets/MOOCRadar/test_triples_0_639.csv"  # 请根据实际路径修改
    output_file_path = "student_answer_probability_report.txt"
    
    # 检查CSV文件是否存在
    if not Path(csv_file_path).exists():
        print(f"错误: CSV文件不存在: {csv_file_path}")
        print("请检查文件路径是否正确")
        return
    
    # 执行计算
    result = calculate_answer_probability(csv_file_path, output_file_path)
    
    if result:
        print("\n计算完成!")
    else:
        print("\n计算失败!")

if __name__ == "__main__":
    main() 