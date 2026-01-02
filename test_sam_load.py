import torch
from segment_anything import sam_model_registry

ckpt = "checkpoints/sam_vit_h_4b8939.pth"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading SAM...")
sam = sam_model_registry["vit_h"](checkpoint=ckpt)
sam.to(device=device)
sam.eval()
print("Loaded SAM to", device)
