import torch
import torch.nn as nn

from diffusiontools.analysis import plot_image_diffusion
from diffusiontools.models import CNN, DDPM

# Set random seed for greater reproducibility.
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(0, 500, 100):
    # Load models.
    ckpt_0001 = torch.load(
        f"data/DDPM/checkpoint/ddpm_checkpoint_0001_{epoch:04d}.pt",
        map_location=device,
    )
    ckpt_0002 = torch.load(
        f"data/DDPM/checkpoint/ddpm_checkpoint_0002_{epoch:04d}.pt",
        map_location=device,
    )

    gt_1 = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 32, 16),
        act=nn.GELU,
    ).to(device)
    ddpm_1 = DDPM(gt=gt_1, betas=(0.0001, 0.02), n_T=1000).to(device)
    ddpm_1.load_state_dict(ckpt_0001["model_state_dict"])

    gt_2 = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 32, 16),
        act=nn.GELU,
    ).to(device)
    ddpm_2 = DDPM(gt=gt_2, betas=(0.001, 0.2), n_T=100).to(device)
    ddpm_2.load_state_dict(ckpt_0002["model_state_dict"])

    # Create diffusion plots.
    ddpm_1.eval()
    with torch.no_grad():
        plot_image_diffusion(
            ddpm_1,
            [1.0, 0.1, 0.0],
            10,
            (1, 28, 28),
            device,
            f"Intermediate Diffusion Images for Model 1 at Epoch {epoch}",
            f"diffusion_plot_1_{epoch:04d}.png",
        )

    ddpm_2.eval()
    with torch.no_grad():
        plot_image_diffusion(
            ddpm_2,
            [1.0, 0.1, 0.0],
            10,
            (1, 28, 28),
            device,
            f"Intermediate Diffusion Images for Model 2 at Epoch {epoch}",
            f"diffusion_plot_2_{epoch:04d}.png",
        )
