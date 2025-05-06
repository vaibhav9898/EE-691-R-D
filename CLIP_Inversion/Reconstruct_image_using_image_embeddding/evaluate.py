# evaluate.py

import torch
import lpips
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim
import numpy as np

lpips_model = lpips.LPIPS(net='alex')  # or 'vgg'

def compute_lpips(img1, img2):
    device = next(lpips_model.parameters()).device
    img1 = img1.to(device)
    img2 = img2.to(device)
    return lpips_model(img1, img2).item()

def compute_ssim(img1, img2):
    # Convert from tensor to numpy
    img1_np = img1.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img2_np = img2.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    return ssim(img1_np, img2_np, channel_axis=2, data_range=1.0)

def compute_clip_cosine(clip_model, img1, img2, device):
    preprocess = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    img1_norm = preprocess(img1.squeeze()).unsqueeze(0).to(device)
    img2_norm = preprocess(img2.squeeze()).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = clip_model.encode_image(img1_norm)
        emb2 = clip_model.encode_image(img2_norm)
        cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    return cosine_sim
