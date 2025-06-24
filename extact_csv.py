import pandas as pd

def extract_first_640_rows(input_file: str, output_file: str):
    """
    从parquet文件中提取前640条数据并保存
    
    Args:
        input_file: 输入parquet文件路径
        output_file: 输出parquet文件路径
    """
    try:
        # 读取parquet文件
        df = pd.read_parquet(input_file)
        
        # 提取前640行数据
        filtered_df = df.head(640)
        
        # 保存到新的parquet文件
        filtered_df.to_parquet(output_file)
        
        print(f"成功提取数据并保存至 {output_file}")
        print(f"提取的数据条数: {len(filtered_df)}")
        print(f"原始数据条数: {len(df)}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

# 使用示例
input_file = 'datasets/MOOCRadar/trained_data.parquet'  # 输入文件路径
output_file = 'datasets/MOOCRadar/trained_data_0_639.parquet'  # 输出文件路径

extract_first_640_rows(input_file, output_file)