import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from utils import (hook_function0, hook_function1, kl_divergence,
                  cosine_similarity_loss, feature_orthogonality_loss,
                  quantization_constraint, features0, features1)

def train(csf, gen, train_loader, fixed_noise, fixed_input_pdf, cfg, device):
    # Initialize optimizer
    gen_optimizer = optim.Adam(gen.parameters(), lr=0.0005, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(cfg.num_epochs):
        gen_optimizer.zero_grad()

        # Generate input
        noise = torch.randn(cfg.gen_batch_size, cfg.nz, 1, 1).to(device)
        input_pdf = F.softmax(torch.randn(cfg.gen_batch_size, cfg.n_classes).to(device), dim=1)
        set_labels = torch.argmax(input_pdf, dim=1)

        # Generate images
        gen_images = gen(noise, input_pdf)
        quant_loss = quantization_constraint(gen_images)

        # Register hooks
        hook0 = csf.main[8].register_forward_hook(hook_function0)
        hook1 = csf.main[5].register_forward_hook(hook_function1)

        # Forward pass through classifier
        csf_outputs = csf(gen_images)
        output_pdf = F.softmax(csf_outputs, dim=1)

        # Compute losses
        kl_divergence_value = kl_divergence(output_pdf, input_pdf)
        cosine_similarity = cosine_similarity_loss(features0[0]) + cosine_similarity_loss(features1[0])
        orthogonality_loss = feature_orthogonality_loss(features0[0]) + feature_orthogonality_loss(features1[0])
        cross_entropy = criterion(csf_outputs, set_labels)
        predicted_labels = torch.argmax(F.log_softmax(csf_outputs, dim=1), dim=1)
        accuracy = (predicted_labels == set_labels).float().mean()

        l1_regularization = sum(param.abs().sum() for param in gen.parameters())

        # Combined loss
        loss = (
            10 * cosine_similarity +
            0.001 * orthogonality_loss +
            10 * cross_entropy +
            cfg.lambda_l1 * l1_regularization
        )

        # Backward pass
        loss.backward()
        gen_optimizer.step()

        # Clean up
        hook0.remove()
        hook1.remove()
        features0.clear()
        features1.clear()

        # Print progress
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}, "
             ,  f"KLD:{kl_divergence_value.item():.4f}, "
              f"CE: {cross_entropy.item():.4f}, OL: {orthogonality_loss.item():.4f}, "
              f"CS: {cosine_similarity.item():.4f}, IA: {accuracy.item():.4f}")

        # Save generated images every 100 epochs
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                fixed_gen_images = gen(fixed_noise, fixed_input_pdf)
                vutils.save_image(
                    fixed_gen_images, f"MNIST-Inversion/1/image_{epoch + 1}.png",
                    nrow=10, normalize=True
                )