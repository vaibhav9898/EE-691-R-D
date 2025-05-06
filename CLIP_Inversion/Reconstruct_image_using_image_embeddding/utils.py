# utils.py

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from PIL import ImageFilter

def load_image(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def initialize_image(method="noise", size=224, device="cuda", original_img=None):
    if method == "noise":
        img = torch.randn(1, 3, size, size, device=device)
    elif method == "gray":
        img = torch.full((1, 3, size, size), 0.5, device=device)
    elif method == "blur":
        if original_img is None:
            raise ValueError("original_img must be provided for 'blur'")
        pil_blurred = TF.to_pil_image(original_img.squeeze().cpu()).filter(ImageFilter.GaussianBlur(radius=5))
        tensor_blurred = TF.to_tensor(pil_blurred).unsqueeze(0).to(device)
        img = tensor_blurred.clone().detach()
    else:
        raise ValueError(f"Unknown init method: {method}")
    img.requires_grad = True
    return img


def deprocess_image(img_tensor):
    img = img_tensor.detach().cpu().squeeze().clamp(0, 1)
    img = transforms.ToPILImage()(img)
    return img
