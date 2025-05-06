# losses.py

import torch
import torch.nn.functional as F

def clip_embedding_loss(embedding_pred, embedding_target):
    return F.mse_loss(embedding_pred, embedding_target)

def total_variation_loss(img):
    loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return loss

def perceptual_loss(vgg, recon_img, target_img):
    recon_feats = vgg(recon_img)
    target_feats = vgg(target_img)
    loss = 0.0
    for rf, tf in zip(recon_feats, target_feats):
        loss += F.mse_loss(rf, tf)
    return loss

def cosine_similarity_loss(embedding_pred, embedding_target):
    cosine_sim = F.cosine_similarity(embedding_pred, embedding_target, dim=-1)
    return 1 - cosine_sim.mean()  # 1 - cosθ → minimize when cosθ ≈ 1

