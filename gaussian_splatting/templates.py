import os
import torch 
from simple_trainer import Config, Runner

def load_gs_ckpt(path_ckpt : str, device="cuda"):
    """ Function to read a checkpoint 
        input : directory to the checkpoint
        output : dictionary with the splats """
    
    ckpt = torch.load(path_ckpt, map_location=device, weights_only=True)
    return ckpt

dataset = "hope"
object_id = 1
root_ckpts = "/home/sergio/onboarding_stage/gaussian_splatting/results"
n_iters = 15000
local_rank = 0
ckpt_name = f"ckpt_{n_iters-1}_rank{local_rank}.pt"
path_ckpt = f"{root_ckpts}/{dataset}/obj_{object_id:06d}/ckpts/{ckpt_name}"
model_1 = load_gs_ckpt(path_ckpt)
print("loaded model")