import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import shutil

def merge_model_weights(weights_dir: str, output_dir: str):
    """
    合并分布式训练的模型权重文件
    
    Args:
        weights_dir: 包含分布式权重文件的目录
        output_dir: 输出合并后权重的目录
    """
    print("开始合并模型权重...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 首先尝试加载模型
    try:
        print("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            weights_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 保存合并后的模型
        print("正在保存合并后的模型...")
        model.save_pretrained(output_dir)
        
        # 复制tokenizer文件
        print("正在复制tokenizer文件...")
        tokenizer = AutoTokenizer.from_pretrained(weights_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"模型已成功合并并保存到: {output_dir}")
        
    except Exception as e:
        print(f"使用AutoModel加载失败: {str(e)}")
        print("尝试使用备用方法...")
        
        try:
            # 备用方法：直接合并权重文件
            print("正在合并权重文件...")
            model_weights = {}
            
            # 遍历所有rank的权重文件
            for rank in range(2):
                model_path = os.path.join(weights_dir, f"model_world_size_2_rank_{rank}.pt")
                if os.path.exists(model_path):
                    print(f"加载rank {rank}的权重...")
                    state_dict = torch.load(model_path, map_location='cpu')
                    for key, value in state_dict.items():
                        if key not in model_weights:
                            model_weights[key] = value
                        else:
                            # 如果键已存在，尝试合并张量
                            if isinstance(value, torch.Tensor) and isinstance(model_weights[key], torch.Tensor):
                                try:
                                    model_weights[key] = torch.cat([model_weights[key], value], dim=0)
                                except:
                                    print(f"警告：无法合并张量 {key}，使用第一个值")
                            else:
                                print(f"警告：键 {key} 不是张量，使用第一个值")
            
            # 保存合并后的权重
            print("正在保存合并后的权重...")
            torch.save(model_weights, os.path.join(output_dir, "pytorch_model.bin"))
            print(f"权重已成功合并并保存到: {output_dir}")
            
        except Exception as e:
            print(f"备用方法也失败了: {str(e)}")
            raise

if __name__ == "__main__":
    # 设置输入和输出目录
    weights_dir = "/d2/mxy/cat-r1/cat-r1/checkpoints/verl_examples/gsm8k/global_step_5/actor"
    output_dir = "/d2/mxy/cat-r1/cat-r1/checkpoints/verl_examples/gsm8k/global_step_5/merged_hf_model"
    
    # 执行合并
    merge_model_weights(weights_dir, output_dir)