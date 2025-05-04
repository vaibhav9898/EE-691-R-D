import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, n_classes):
        super(Generator, self).__init__()
        # Linear layer to map the softmaxed vector to the size nz
        self.embed = nn.Linear(n_classes, nz)

        # The main model architecture remains unchanged
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, ngf * 8, 4, 1, 0, bias=False),   # 1 → 4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 4 → 8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 8 → 16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),      # 16 → 32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),           # 32 → 64
            nn.Tanh()
        )

    def forward(self, z, class_vector):
        # Map the class vector to the latent space size
        embed_vector = self.embed(class_vector).unsqueeze(-1).unsqueeze(-1)
        # Concatenate with the latent vector z
        input = torch.cat([z, embed_vector], 1)
        return self.main(input)

class Classifier(nn.Module):
    def __init__(self, nc, ncf):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ncf, 4, 2, 1, bias=False),      # 64 → 32
            nn.BatchNorm2d(ncf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ncf, ncf * 2, 4, 2, 1, bias=False),  # 32 → 16
            nn.BatchNorm2d(ncf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ncf * 2, ncf * 4, 4, 2, 1, bias=False),  # 16 → 8
            nn.BatchNorm2d(ncf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),                                     # (ncf*4, 8, 8) → (ncf*4*8*8)
            nn.Linear(ncf * 4 * 8 * 8, ncf * 2),
            nn.ReLU(),

            nn.Linear(ncf * 2, 6)
        )

    def forward(self, x):
        return self.main(x)