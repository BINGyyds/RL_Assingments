import torch

# 检查 CUDA 是否可用
print(torch.cuda.is_available())

# 如果可用，查看 GPU 数量
print(torch.cuda.device_count())

# 查看当前 GPU 的名称
print(torch.cuda.get_device_name(0))