import torch
import sys
import os

# 导入模型
from regression_models import Reg_flat_iid

def test_reg_flat_iid():
    # 模型参数
    dim_data = 10  # 输入数据维度
    dim_output = 2  # 输出维度
    dim_middle = 64  # 中间层维度
    agg_num_heads = 4  # 注意力头数
    agg_num_layers = 2  # Transformer层数
    
    # 创建模型实例
    model = Reg_flat_iid(
        dim_data=dim_data,
        dim_output=dim_output,
        dim_middle=dim_middle,
        agg_num_heads=agg_num_heads,
        agg_num_layers=agg_num_layers
    )
    
    # 设置为评估模式
    model.eval()
    
    # 创建测试数据
    batch_size = 3
    seq_len = 5
    x = torch.randn(batch_size, seq_len, dim_data)
    
    # 创建掩码 (假设所有序列都是有效的)
    mask = torch.ones(batch_size, seq_len)
    
    # 部分序列长度较短，设置掩码
    mask[0, 3:] = 0  # 第一个样本只有3个有效序列
    mask[1, 4:] = 0  # 第二个样本只有4个有效序列
    
    # 前向传播
    print("开始前向传播...")
    with torch.no_grad():
        output = model(x, mask)
    
    # 检查输出形状
    expected_shape = (batch_size, dim_output)
    actual_shape = output.shape
    
    print(f"期望输出形状: {expected_shape}")
    print(f"实际输出形状: {actual_shape}")
    print(f"测试{'通过' if expected_shape == actual_shape else '失败'}")
    
    if expected_shape == actual_shape:
        print("模型前向传播正常!")
        print("输出示例:")
        print(output)
    
if __name__ == "__main__":
    test_reg_flat_iid() 