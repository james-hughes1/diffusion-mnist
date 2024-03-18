"""!@file main.py
    @brief Runs DDPM diffusion model.
    @author Created by J. Hughes on 18/03/2024.
"""

from diffusiontools.models import CNN, DDPM
from diffusiontools.train import train_ddpm

import sys
import configparser as cfg
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# Load hyperparameters.
config = cfg.ConfigParser()

if len(sys.argv) == 2:
    input_file = sys.argv[1]
    config.read(input_file)
else:
    print("No config file given, using default hyperparameters.")

beta1 = config.getfloat("model", "beta1", fallback=1e-4)
beta2 = config.getfloat("model", "beta2", fallback=0.02)
n_T = config.getint("model", "n_T", fallback=1000)
n_hidden = config.get("model", "n_hidden", fallback="16 32 32 16")
n_hidden = tuple(int(h) for h in n_hidden.split())

n_epoch = config.getint("training", "n_epoch", fallback=100)
batch_size = config.getint("training", "batch_size", fallback=128)
lr_initial = config.getfloat("training", "lr_initial", fallback=2e-4)

model_path = config.get("output", "model_path", fallback="./data/DDPM/model/")
sample_path = config.get(
    "output", "sample_path", fallback="./data/DDPM/sample/"
)
save_interval = config.getint("output", "save_interval", fallback=10)
config_id = config.getint("output", "config_id", fallback=1234)

# Load MNIST Data with pre-processing.
tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
)
dataset = MNIST("./data", train=True, download=True, transform=tf)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
)

# Create DDPM model.
gt = CNN(
    in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU
)
ddpm = DDPM(gt=gt, betas=(beta1, beta2), n_T=n_T)
optim = torch.optim.Adam(ddpm.parameters(), lr=lr_initial)

accelerator = Accelerator()

# Initialise Accelerator object.
ddpm, optim, dataloader = accelerator.prepare(ddpm, optim, dataloader)
print(f"Accelerator initialised, using device {accelerator.device}.")

# Train and save DDPM model.
train_ddpm(
    ddpm,
    optim,
    dataloader,
    accelerator,
    n_epoch,
    save_interval,
    sample_path,
    model_path,
    config_id,
)
