import torch

base_model_path = './trained_model/vgg16/cityscape/bdd100k_ciconv.pth'
state_dict = torch.load(base_model_path)
print(state_dict)