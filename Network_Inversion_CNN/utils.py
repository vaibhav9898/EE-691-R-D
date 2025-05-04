import torch
import torch.nn as nn
import torch.nn.functional as F

# Global lists to store features from hooks
features0 = []
features1 = []

def weights_initialization_gen_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def hook_function0(module, input, output):
    features0.append(output)

def hook_function1(module, input, output):
    features1.append(output)

def generate_sorted_input_pdf(num_samples, n_classes, nz):
    fixed_noise = torch.randn(num_samples, nz, 1, 1)
    fixed_input_pdf = torch.zeros(num_samples, n_classes)
    for i in range(num_samples):
        class_idx = (i // (num_samples // n_classes)) % n_classes
        fixed_input_pdf[i, class_idx] = 1.0
    fixed_input_pdf = F.softmax(fixed_input_pdf, dim=1)
    return fixed_noise, fixed_input_pdf

def quantization_constraint(images):
    # Placeholder for quantization constraint (implement as needed)
    return torch.tensor(0.0, device=images.device)

def kl_divergence(output_pdf, input_pdf):
    return F.kl_div(F.log_softmax(output_pdf, dim=1), input_pdf, reduction='batchmean')

def cosine_similarity_loss(features):
    # Flatten spatial dimensions: [batch_size, channels, height, width] -> [batch_size, channels*height*width]
    batch_size = features.size(0)
    features = features.view(batch_size, -1)
    # Normalize features
    features = F.normalize(features, dim=1)
    # Compute cosine similarity matrix
    cos_sim = torch.matmul(features, features.transpose(0, 1))
    # Exclude diagonal (self-similarity)
    mask = torch.eye(batch_size, device=cos_sim.device).bool()
    cos_sim = cos_sim.masked_fill(mask, 0)
    # Average non-diagonal elements
    return cos_sim.abs().mean()

def feature_orthogonality_loss(features):
    # Flatten spatial dimensions: [batch_size, channels, height, width] -> [batch_size, channels*height*width]
    batch_size = features.size(0)
    features = features.view(batch_size, -1)
    # Compute Gram matrix
    gram = torch.matmul(features, features.transpose(0, 1))
    # Target identity matrix
    identity = torch.eye(batch_size, device=gram.device)
    # Frobenius norm of difference
    return ((gram - identity) ** 2).mean()