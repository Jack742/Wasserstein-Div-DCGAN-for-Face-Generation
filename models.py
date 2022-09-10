import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def layer(in_feat, out_feat, bnorm=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if bnorm:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.SiLU())
            return layers

        self.model = nn.Sequential(
            *layer(latent_dim, 128, bnorm=False),
            *layer(128, 256),
            *layer(256, 512),
            *layer(512, 1024),
            *layer(1024,1556),
            nn.Linear(1556, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 1024),
            nn.SiLU(),
            nn.Linear(1024,512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        score = self.model(img_flat)
        return score