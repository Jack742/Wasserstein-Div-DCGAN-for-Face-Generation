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
            *layer(1024,1256),
            nn.Linear(1256, int(np.prod(img_shape))),
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
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        score = self.model(img_flat)
        return score




# TODO Complete DeepConv code
class DCGenerator(nn.Module):
    def __init__(self,latent_dim):
        super(DCGenerator, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 4*4*1024),
            nn.BatchNorm1d(4*4*1024),
            nn.SiLU(),#Reshape in forward pass
            )

        #self.conv_blocks = nn.Sequential(
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(1024,512,kernel_size=5, stride=2, padding=2,output_padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),

            nn.ConvTranspose2d(512,256,kernel_size=5, stride=2, padding=2,output_padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),

            nn.ConvTranspose2d(256,128,kernel_size=3, stride=2,padding=1,output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            nn.ConvTranspose2d(128,3,kernel_size=3, stride=2, padding=1,output_padding=1, bias=False),
            nn.Tanh()
        )
        

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0],1024,4,4)
        img = self.conv_blocks(out)
        return img


class DCDiscriminator(nn.Module):
    def __init__(self,img_shape):
        super(DCDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, **kwargs):
            block = [nn.Conv2d(in_filters, out_filters, **kwargs)]#, nn.Dropout2d(0.25)
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            
            block.append(nn.SiLU())
            return block


        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 64, bn=False, **{'padding':1,'bias':False, 'stride':2,'kernel_size':5}),
            *discriminator_block(64, 128, **{'bias':False, 'stride':2,'kernel_size':3}),
            *discriminator_block(128, 256, **{'padding':1,'bias':False, 'stride':2,'kernel_size':5}),
            *discriminator_block(256, 512,  **{'padding':2,'bias':False, 'stride':2,'kernel_size':5}),
            nn.Conv2d(512, 1, bias=False, stride=1,kernel_size=4)
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return out