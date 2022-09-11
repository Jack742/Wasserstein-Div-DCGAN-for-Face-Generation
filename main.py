import pprint
import numpy as np

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from models import Generator, Discriminator, DCGenerator, DCDiscriminator
import utils


config = utils.read_config('config.yaml')
pp = pprint.PrettyPrinter(indent=4)
print("#############\nCONFIG LOADED\n#############\n")
pp.pprint(config)
print('\n')

img_shape = (config['img_channels'], config['img_size'], config['img_size'])

cuda = True if torch.cuda.is_available() else False
config['cuda'] = cuda

k = config['wdiv_k_value']
p = config['wdiv_p_value']


# Initialize generator and discriminator
if config['use_conv']:
    print('Using ConvNet')
    generator = DCGenerator(config['latent_dim'])
    discriminator = DCDiscriminator(img_shape)
else:
    generator = Generator(config['latent_dim'], img_shape)
    discriminator = Discriminator(img_shape)

if cuda:
    print('Using CUDA\n')
    generator.cuda()
    discriminator.cuda()

# Configure data loader
dataloader = utils.get_dataloader(config['datafolder_path'], config['img_size'], config['batch_size'])


# optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor






# ----------
#  Training
# ----------

epoch_start = 0
batches_done = 0

if config['load_last_checkpoint']:
    #Load model weights and set epoch start and batches done to correct number
    epoch_start = utils.load_model_weights(generator, discriminator, \
        config['gen_weight_path'],config['disc_weight_path'], config['sample_interval'])
    batches_done = (epoch_start*len(dataloader)) - ((epoch_start*len(dataloader)) % config['sample_interval'])
    print(batches_done)




for epoch in range(epoch_start,config['n_epochs']):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor), requires_grad=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], config['latent_dim']))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)

        # Compute W-div gradient penalty
        real_grad_out = Variable(Tensor(real_imgs.size(0), 1).fill_(1.0),\
             requires_grad=False)

        real_grad = autograd.grad(real_validity, real_imgs, real_grad_out, \
                create_graph=True, retain_graph=True, only_inputs=True)[0]

        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        
        fake_grad = autograd.grad(fake_validity, fake_imgs, fake_grad_out, create_graph=True, \
            retain_graph=True, only_inputs=True)[0]

        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % config['n_critic'] == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, config['n_epochs'], i, len(dataloader), d_loss.item(), g_loss.item())
            )

            if epoch % 50 == 0 or epoch ==config['n_epochs']-1:
                torch.save(generator.state_dict(),config['gen_weight_path']+f'gen_weights_{epoch}')
                torch.save(discriminator.state_dict(),config['disc_weight_path']+f'disc_weights_{epoch}')

            if batches_done % config['sample_interval'] == 0:
                print("#####################\nSAVING SAMPLE IMAGES\n"+config['image_save_path']+f"%d.png"% batches_done+"\n#####################\n")
                
                save_image(fake_imgs.data[:25],config['image_save_path']+"%d.png" % batches_done,\
                     nrow=5, normalize=True)

            batches_done += config['n_critic']
save_image(fake_imgs.data[:25], config['image_save_path']+"%d.png" % batches_done, nrow=5, normalize=True)