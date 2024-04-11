import torch

state_dict = torch.load("dataset/model_best.pt")
torch.save(state_dict, "new_model.pt", _use_new_zipfile_serialization=False)
