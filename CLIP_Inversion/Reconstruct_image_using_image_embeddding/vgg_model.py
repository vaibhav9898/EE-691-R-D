# vgg_model.py

import torch
import torch.nn as nn
from torchvision import models

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=["relu1_2", "relu2_2", "relu3_3"], use_input_norm=True):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.use_input_norm = use_input_norm
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std  = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

        self.layer_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
        }

        max_idx = max(self.layer_map[layer] for layer in layers)
        self.features = nn.Sequential(*[vgg19[i] for i in range(max_idx + 1)])
        self.selected_layers = [self.layer_map[l] for l in layers]

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.selected_layers:
                feats.append(x)
        return feats
