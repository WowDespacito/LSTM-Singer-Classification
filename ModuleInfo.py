import torch
import torchvision.models as models

# 加载预训练模型
model = models.resnet18(pretrained = True)

# 获取模型参数
params = model.state_dict()

# 输出网络信息
for key, value in model.named_parameters():
    print(key, "\t", value.size())