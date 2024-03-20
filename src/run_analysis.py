import torch
import torch.nn as nn

from diffusiontools.analysis import (
    plot_image_diffusion,
    plot_learning_curve,
    compute_image_diffusion,
    compute_variance,
    plot_tsne,
)
from diffusiontools.models import CNN, DDPM

# Set random seed for greater reproducibility.
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models.
ckpt_0001_0490 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0001_0490.pt", map_location=device
)
ckpt_0002_0490 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0002_0490.pt", map_location=device
)

gt_1 = CNN(
    in_channels=1,
    expected_shape=(28, 28),
    n_hidden=(16, 32, 32, 16),
    act=nn.GELU,
).to(device)
ddpm_1 = DDPM(gt=gt_1, betas=(0.0001, 0.02), n_T=1000).to(device)
ddpm_1.load_state_dict(ckpt_0001_0490["model_state_dict"])

gt_2 = CNN(
    in_channels=1,
    expected_shape=(28, 28),
    n_hidden=(16, 32, 32, 16),
    act=nn.GELU,
).to(device)
ddpm_2 = DDPM(gt=gt_2, betas=(0.001, 0.2), n_T=100).to(device)
ddpm_2.load_state_dict(ckpt_0002_0490["model_state_dict"])

# Evaluate models.

SAMPLE_SIZE = 1000

print(f"Sampling {SAMPLE_SIZE} images from each model.\n")

ddpm_1.eval()
with torch.no_grad():
    plot_image_diffusion(
        ddpm_1,
        [1.0, 0.5, 0.1, 0.05, 0.0],
        5,
        (1, 28, 28),
        device,
        "Intermediate Diffusion Images for Final Model 1",
        "diffusion_plot_1.png",
    )
    _, sample1 = compute_image_diffusion(
        ddpm_1, [0.0], SAMPLE_SIZE, (1, 28, 28), device
    )

ddpm_2.eval()
with torch.no_grad():
    plot_image_diffusion(
        ddpm_2,
        [1.0, 0.5, 0.1, 0.05, 0.0],
        5,
        (1, 28, 28),
        device,
        "Intermediate Diffusion Images for Final Model 2",
        "diffusion_plot_2.png",
    )
    _, sample2 = compute_image_diffusion(
        ddpm_2, [0.0], SAMPLE_SIZE, (1, 28, 28), device
    )

print("Model 1 Statistics:")
plot_learning_curve(
    ckpt_0001_0490["losses_train"],
    ckpt_0001_0490["losses_val"],
    "Learning Curve for DDPM Config. 1",
    "learning_curve_1.png",
)
total_var_1 = compute_variance(sample1[0])
print(f"Total variance of model 1 is {total_var_1}\n")

print("Model 2 Statistics:")
plot_learning_curve(
    ckpt_0002_0490["losses_train"],
    ckpt_0002_0490["losses_val"],
    "Learning Curve for DDPM Config. 2",
    "learning_curve_2.png",
)
total_var_2 = compute_variance(sample2[0])
print(f"Total variance of model 2 is {total_var_2}")

# Plot joint t-SNE plot in 2D.
plot_tsne(sample1[0], sample2[0], "tsne_mnist.png")
