# import torch
# import clip
# import os

# from config import CONFIG
# from utils import load_image, initialize_image, deprocess_image
# from losses import clip_embedding_loss, total_variation_loss, perceptual_loss, cosine_similarity_loss
# from vgg_model import VGGFeatureExtractor
# from evaluate import compute_lpips, compute_ssim, compute_clip_cosine

# # Load CLIP model
# device = CONFIG["device"]
# model, preprocess = clip.load(CONFIG["model_name"], device=device)

# # Load target image
# img = load_image(CONFIG["image_path"], preprocess).to(device)

# # Setup VGG perceptual model (only needed for optimization)
# vgg_model = VGGFeatureExtractor().to(device).eval()

# frames=[]


# def run_optimization(img, model, vgg_model, variant_id=0):
#     print("[INFO] Running optimization-based inversion...")

#     # Get target CLIP embedding
#     with torch.no_grad():
#         target_embedding = model.encode_image(img).detach()

#     # Initialize reconstruction
#     recon_img = initialize_image(CONFIG["init_method"], CONFIG["img_size"], device, original_img=img)

#     # Optimizer
#     optimizer = torch.optim.Adam([recon_img], lr=CONFIG["lr"])

#     # Loss weights
#     tv_weight = 1e-6
#     perc_weight = 1.0

#     for step in range(CONFIG["num_steps"]):
#         optimizer.zero_grad()

#         normed_input = torch.nn.functional.normalize(
#             (recon_img - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)) /
#             torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1),
#             dim=1
#         )

#         pred_embedding = model.encode_image(normed_input)
#         loss_clip = clip_embedding_loss(pred_embedding, target_embedding)
#         loss_tv = total_variation_loss(recon_img)
#         loss_perc = perceptual_loss(vgg_model, recon_img, img)
#         loss_cosine = cosine_similarity_loss(pred_embedding, target_embedding)

#         loss = loss_clip + tv_weight * loss_tv + perc_weight * loss_perc + 0.5*loss_cosine
#         loss.backward()
#         optimizer.step()

#         with torch.no_grad():
#             recon_img.clamp_(0, 1)

#         if step % CONFIG["save_every"] == 0 or step == CONFIG["num_steps"] - 1:
#             print(f"Step {step}, Loss: {loss.item():.4f} (clip: {loss_clip.item():.4f}, TV: {loss_tv.item():.4f}, perc: {loss_perc.item():.4f}), cosine: {loss_cosine.item():.4f}")
#             img_out = deprocess_image(recon_img)
#             frames.append(img_out.copy())
#             os.makedirs("outputs", exist_ok=True)
#             img_out.save(f"outputs/reconstructed_final_{step}.png")

#     # Final evaluation
#     lp = compute_lpips(recon_img, img)
#     ss = compute_ssim(recon_img, img)
#     cs = compute_clip_cosine(model, recon_img, img, device)

#     print(f"[Final Eval] LPIPS: {lp:.4f} | SSIM: {ss:.4f} | CLIP Cosine: {cs:.4f}")


# # ==== MAIN ====

# if CONFIG["method"] == "optimization":
#     run_optimization(img, model, vgg_model)
#     if len(frames) > 1:
#         frames[0].save(f"animations/reconstruction_v{variant_id}.gif", save_all=True,
#                    append_images=frames[1:], duration=300, loop=0)





import torch
import clip
import os
from config import CONFIG
from utils import load_image, initialize_image, deprocess_image
from losses import (
    clip_embedding_loss,
    total_variation_loss,
    perceptual_loss,
    cosine_similarity_loss
)
from vgg_model import VGGFeatureExtractor
from evaluate import compute_lpips, compute_ssim, compute_clip_cosine


# Load CLIP model
device = CONFIG["device"]
model, preprocess = clip.load(CONFIG["model_name"], device=device)

# Load target image
img = load_image(CONFIG["image_path"], preprocess).to(device)

# Setup VGG perceptual model
vgg_model = VGGFeatureExtractor().to(device).eval()


def run_optimization(img, model, vgg_model, variant_id=0):
    print("[INFO] Running optimization-based inversion...")

    # Get target CLIP embedding
    with torch.no_grad():
        target_embedding = model.encode_image(img).detach()

    # Initialize reconstruction image
    recon_img = initialize_image(CONFIG["init_method"], CONFIG["img_size"], device, original_img=img)

    # Optimizer
    optimizer = torch.optim.Adam([recon_img], lr=CONFIG["lr"])

    # Loss weights
    tv_weight = 1e-6
    perc_weight = 1.0
    cosine_weight = CONFIG.get("cosine_loss_weight", 0.5)

    # For animation
    frames = []

    for step in range(CONFIG["num_steps"]):
        optimizer.zero_grad()

        # Normalize for CLIP
        normed_input = torch.nn.functional.normalize(
            (recon_img - torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)) /
            torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1),
            dim=1
        )

        pred_embedding = model.encode_image(normed_input)
        loss_clip = clip_embedding_loss(pred_embedding, target_embedding)
        loss_tv = total_variation_loss(recon_img)
        loss_perc = perceptual_loss(vgg_model, recon_img, img)

        loss = loss_clip + tv_weight * loss_tv + perc_weight * loss_perc

        if CONFIG.get("use_cosine_loss", False):
            loss_cosine = cosine_similarity_loss(pred_embedding, target_embedding)
            loss += cosine_weight * loss_cosine
        else:
            loss_cosine = torch.tensor(0.0)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            recon_img.clamp_(0, 1)

        # Save intermediate step and store for GIF
        if step % CONFIG["save_every"] == 0 or step == CONFIG["num_steps"] - 1:
            print(f"Step {step}, Loss: {loss.item():.4f} "
                  f"(clip: {loss_clip.item():.4f} ,"
                  f"TV: {loss_tv.item():.4f}, perc: {loss_perc.item():.4f})")

            img_out = deprocess_image(recon_img)
            os.makedirs("outputs", exist_ok=True)
            img_out.save(f"outputs/reconstructed_v{variant_id}_step{step}.png")
            if CONFIG.get("save_gif", False):
                frames.append(img_out.copy())

    # Final evaluation
    lp = compute_lpips(recon_img, img)
    ss = compute_ssim(recon_img, img)
    cs = compute_clip_cosine(model, recon_img, img, device)
    print(f"[Final Eval - Variant {variant_id}] LPIPS: {lp:.4f} | SSIM: {ss:.4f} | CLIP Cosine: {cs:.4f}")

    # Save GIF
    if CONFIG.get("save_gif", False) and len(frames) > 1:
        os.makedirs("animations", exist_ok=True)
        frames[0].save(
            f"animations/reconstruction_v{variant_id}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=CONFIG.get("gif_duration", 300),
            loop=0
        )


# ==== MAIN ====
if CONFIG["method"] == "optimization":
    for i in range(CONFIG.get("num_variants", 1)):
        print(f"\n[Variant {i + 1}/{CONFIG['num_variants']}]")
        run_optimization(img, model, vgg_model, variant_id=i)
else:
    raise ValueError(f"Unknown method {CONFIG['method']}")
