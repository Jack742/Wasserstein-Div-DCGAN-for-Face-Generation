import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms



def read_config(file_path: str) -> dict:
    """
    Loads YAML Config file from provided path
    || file_path |-> path to config file
    || returns: Dictionary object containing config parameters
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
        return config

def get_dataloader(datafolder_path: str, img_size: int, batch_size: int) -> DataLoader:
    """
    Loads data from path to an ImageFolder |-> DataLoader
    || img_size |-> dimensions to resize image to
    || batch_size |-> size of batches for dataloader to process
    || returns DataLoader object
    """
    return DataLoader(
    datasets.ImageFolder(
        datafolder_path,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(0.5),transforms.Resize((img_size,img_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

def load_model_weights(generator: nn.Module, discriminator: nn.Module, gen_weight_path: str, disc_weight_path: str, sample_interval: int) -> None:
    """
    loads model weights from file for generator and discriminator models
    || generator |-> generator model
    || discriminator |-> discriminator model
    || gen_weight_path |-> path to generator weights folder
    || disc_weight_path |-> path to discriminator weights folder
    || sample_interval |-> interval between image sampling. Used to calculate batches done so far
    || returns None
    """
    weights = sorted([each for each in os.listdir(gen_weight_path)], key=lambda x: int(x.split('_')[-1]))[-1]
    epoch_start = int(weights.split('_')[-1])
    batches_done = epoch_start * sample_interval
    
    print(f"###################################\nWEIGHTS_LOADED\nEpoch:{epoch_start}\nBatches Done: {batches_done}\n")

    generator.load_state_dict(torch.load(f"{gen_weight_path}/{weights}"))

    weights = sorted([each for each in os.listdir(disc_weight_path) if 'disc' in each], key=lambda x: int(x.split('_')[-1]))[-1]
    discriminator.load_state_dict(torch.load(f"{disc_weight_path}/{weights}"))


