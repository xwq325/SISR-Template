import torch
# if your model train with PyTorch >= 1.6, and test with PyTorch < 1.6, use tran.py with train environment.
state_dict = torch.load("dataset/model_best.pt")
torch.save(state_dict, "new_model.pt", _use_new_zipfile_serialization=False)
