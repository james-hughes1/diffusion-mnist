"""!@file main.py
    @brief Used to create and train model .
    @author Created by J. Hughes on 18/03/2024.
"""

from diffusiontools.models import CNN, DDPM, DMCustom
from diffusiontools.train import train_model

import sys
import configparser as cfg
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# Set random seed for greater reproducibility.
torch.manual_seed(42)

# Load hyperparameters.
config = cfg.ConfigParser()

if len(sys.argv) == 2:
    input_file = sys.argv[1]
    config.read(input_file)
else:
    print("No config file given, using default hyperparameters.")

# Read config hyperparamters.
noise_min = config.getfloat("model", "noise_min", fallback=1e-4)
noise_max = config.getfloat("model", "noise_max", fallback=0.02)
n_T = config.getint("model", "n_T", fallback=1000)
degradation = config.get("model", "degradation", fallback="gaussian")
n_hidden = config.get("model", "n_hidden", fallback="16 32 32 16")
n_hidden = tuple(int(h) for h in n_hidden.split())
loss_fn = config.get("model", "loss_fn", fallback="L2")

n_epoch = config.getint("training", "n_epoch", fallback=100)
batch_size = config.getint("training", "batch_size", fallback=128)
lr_initial = config.getfloat("training", "lr_initial", fallback=2e-4)

checkpoint_path = config.get(
    "output", "checkpoint_path", fallback="./data/model/checkpoint/"
)
sample_path = config.get(
    "output", "sample_path", fallback="./data/model/sample/"
)
save_interval = config.getint("output", "save_interval", fallback=10)
config_id = config.getint("output", "config_id", fallback=1234)

# Load MNIST Data with pre-processing.
tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
)
dataset = MNIST("./data", train=True, download=True, transform=tf)

# Split into train and validation, and create dataloaders
dataset_train, dataset_val = random_split(dataset, [0.8, 0.2])
dataloader_train = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)

# Create model model.
gt = CNN(
    in_channels=1, expected_shape=(28, 28), n_hidden=n_hidden, act=nn.GELU
)

if loss_fn == "L2":
    criterion = nn.MSELoss()
elif loss_fn == "L1":
    criterion = nn.L1Loss()

if degradation == "gaussian":
    model = DDPM(
        gt=gt, betas=(noise_min, noise_max), n_T=n_T, criterion=criterion
    )
elif degradation == "custom":
    model = DMCustom(
        gt=gt,
        alphas=(noise_min, noise_max),
        n_T=n_T,
        size=(1, 28, 28),
        criterion=criterion,
    )
else:
    print(f"Unspecified degradation type {degradation}.")
optim = torch.optim.Adam(model.parameters(), lr=lr_initial)

accelerator = Accelerator()

# Initialise Accelerator object.
model, optim, dataloader_train, dataloader_val = accelerator.prepare(
    model, optim, dataloader_train, dataloader_val
)
print(f"Accelerator initialised, using device {accelerator.device}.")

# Train and save model model.
train_model(
    model,
    optim,
    dataloader_train,
    dataloader_val,
    accelerator,
    n_epoch,
    save_interval,
    sample_path,
    checkpoint_path,
    config_id,
)
