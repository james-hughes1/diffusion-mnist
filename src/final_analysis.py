import torch
import torch.nn as nn

from diffusiontools.analysis import (
    plot_image_diffusion,
    plot_learning_curve,
    compute_image_diffusion,
    compute_variance,
    degradation_demo,
    plot_tsne,
)
from diffusiontools.models import CNN, DDPM, DMCustom

# Set random seed for greater reproducibility.
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models.
gt_1 = CNN(
    in_channels=1,
    expected_shape=(28, 28),
    n_hidden=(16, 32, 32, 16),
    act=nn.GELU,
).to(device)
model_1 = DDPM(gt=gt_1, betas=(0.0001, 0.02), n_T=1000).to(device)

ckpt_0001_0490 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0001_0490.pt", map_location=device
)
model_1.load_state_dict(ckpt_0001_0490["model_state_dict"])

gt_2 = CNN(
    in_channels=1,
    expected_shape=(28, 28),
    n_hidden=(16, 32, 32, 16),
    act=nn.GELU,
).to(device)
model_2 = DDPM(gt=gt_2, betas=(0.001, 0.2), n_T=100).to(device)

ckpt_0002_0490 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0002_0490.pt", map_location=device
)
model_2.load_state_dict(ckpt_0002_0490["model_state_dict"])

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

ckpt_0009_0080 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0009_0080.pt", map_location=device
)
model_9.load_state_dict(ckpt_0009_0080["model_state_dict"])

gt_10 = CNN(
    in_channels=1,
    expected_shape=(28, 28),
    n_hidden=(16, 32, 32, 16),
    act=nn.GELU,
).to(device)
model_10 = DMCustom(
    gt=gt_10, alphas=(0.035, 0.2), n_T=20, size=(1, 28, 28)
).to(device)

ckpt_0010_0120 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0010_0120.pt", map_location=device
)
model_10.load_state_dict(ckpt_0010_0120["model_state_dict"])

plot_learning_curve(
    ckpt_0001_0490["losses_train"],
    ckpt_0001_0490["losses_val"],
    "",
    "learning_curve_1.png",
)
plot_learning_curve(
    ckpt_0002_0490["losses_train"],
    ckpt_0002_0490["losses_val"],
    "",
    "learning_curve_2.png",
)
plot_learning_curve(
    ckpt_0009_0080["losses_train"],
    ckpt_0009_0080["losses_val"],
    "",
    "learning_curve_9.png",
)
plot_learning_curve(
    ckpt_0010_0120["losses_train"],
    ckpt_0010_0120["losses_val"],
    "",
    "learning_curve_10.png",
)

# Demonstration of custom degradation scheme.

degradation_demo(
    model_9, device, [100, 75, 50, 25, 0], "", "custom_degradation.png"
)

# Evaluate models.

SAMPLE_SIZE = 10

print(f"Sampling {SAMPLE_SIZE} images from each model.\n")

model_1.eval()
with torch.no_grad():
    plot_image_diffusion(
        model_1,
        [1.0, 0.5, 0.1, 0.05, 0.0],
        5,
        (1, 28, 28),
        device,
        "Intermediate Diffusion Images for Final Model 1",
        "diffusion_plot_1.png",
    )
    _, sample1 = compute_image_diffusion(
        model_1, [0.0], SAMPLE_SIZE, (1, 28, 28), device
    )

model_2.eval()
with torch.no_grad():
    plot_image_diffusion(
        model_2,
        [1.0, 0.5, 0.1, 0.05, 0.0],
        5,
        (1, 28, 28),
        device,
        "Intermediate Diffusion Images for Final Model 2",
        "diffusion_plot_2.png",
    )
    _, sample2 = compute_image_diffusion(
        model_2, [0.0], SAMPLE_SIZE, (1, 28, 28), device
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
