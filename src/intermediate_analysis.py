import torch
import torch.nn as nn

from diffusiontools.analysis import plot_image_diffusion
from diffusiontools.models import CNN, DDPM, DMCustom

# Set random seed for greater reproducibility.
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(180, 190, 10):
    # Load models.
    ckpt_0001 = torch.load(
        f"data/DDPM/checkpoint/ddpm_checkpoint_0001_{epoch:04d}.pt",
        map_location=device,
    )

    gt_1 = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 32, 16),
        act=nn.GELU,
    ).to(device)
    model_1 = DDPM(gt=gt_1, betas=(0.0001, 0.02), n_T=1000).to(device)
    model_1.load_state_dict(ckpt_0001["model_state_dict"])

    model_1.eval()
    with torch.no_grad():
        plot_image_diffusion(
            model_1,
            [1.0, 0.5, 0.1, 0.05, 0.0],
            5,
            (1, 28, 28),
            device,
            "",
            f"diffusion_plot_1_{epoch:04d}.png",
        )

    ckpt_0002 = torch.load(
        f"data/DDPM/checkpoint/ddpm_checkpoint_0002_{epoch:04d}.pt",
        map_location=device,
    )

    gt_2 = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 32, 16),
        act=nn.GELU,
    ).to(device)
    model_2 = DDPM(gt=gt_2, betas=(0.0001, 0.02), n_T=1000).to(device)
    model_2.load_state_dict(ckpt_0001["model_state_dict"])

    model_2.eval()
    with torch.no_grad():
        plot_image_diffusion(
            model_2,
            [1.0, 0.5, 0.1, 0.05, 0.0],
            5,
            (1, 28, 28),
            device,
            "",
            f"diffusion_plot_2_{epoch:04d}.png",
        )

    if epoch <= 80:
        ckpt_0009 = torch.load(
            f"data/DDPM/checkpoint/ddpm_checkpoint_0009_{epoch:04d}.pt",
            map_location=device,
        )

        gt_9 = CNN(
            in_channels=1,
            expected_shape=(28, 28),
            n_hidden=(16, 32, 32, 16),
            act=nn.GELU,
        ).to(device)
        model_9 = DMCustom(
            gt=gt_9,
            alphas=(0.035, 0.15),
            n_T=100,
            size=(1, 28, 28),
            criterion=nn.L1Loss(),
        ).to(device)
        model_9.load_state_dict(ckpt_0009["model_state_dict"])

        model_9.eval()
        with torch.no_grad():
            plot_image_diffusion(
                model_9,
                [1.0, 0.5, 0.1, 0.05, 0.0],
                5,
                (1, 28, 28),
                device,
                "",
                f"diffusion_plot_9_{epoch:04d}.png",
            )

    ckpt_0010 = torch.load(
        f"data/DDPM/checkpoint/ddpm_checkpoint_0010_{epoch:04d}.pt",
        map_location=device,
    )

    gt_10 = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 32, 16),
        act=nn.GELU,
    ).to(device)
    model_10 = DMCustom(
        gt=gt_10, alphas=(0.035, 0.2), n_T=20, size=(1, 28, 28)
    ).to(device)
    model_10.load_state_dict(ckpt_0010["model_state_dict"])

    # Create diffusion plots.

    model_10.eval()
    with torch.no_grad():
        plot_image_diffusion(
            model_10,
            [1.0, 0.5, 0.2, 0.1, 0.0],
            5,
            (1, 28, 28),
            device,
            "",
            f"diffusion_plot_10_{epoch:04d}.png",
        )
