import torch
import torch.nn as nn

from diffusiontools.analysis import (
    plot_image_diffusion,
    plot_learning_curve,
    compute_image_diffusion,
    compute_image_diffusion_custom,
    compute_variance,
    degradation_demo,
    plot_mnist_tsne,
    compute_tsne_kl_div,
    plot_sample_tsne,
    plot_samples,
    plot_gt_direct_diffusion,
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

ckpt_0010_0180 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0010_0180.pt", map_location=device
)
model_10.load_state_dict(ckpt_0010_0180["model_state_dict"])

# Plot learning curves
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
    ckpt_0010_0180["losses_train"],
    ckpt_0010_0180["losses_val"],
    "",
    "learning_curve_10.png",
)

# Revert to earlier models:
ckpt_0001_0120 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0001_0120.pt", map_location=device
)
model_1.load_state_dict(ckpt_0001_0120["model_state_dict"])

ckpt_0002_0080 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0002_0080.pt", map_location=device
)
model_2.load_state_dict(ckpt_0002_0080["model_state_dict"])

ckpt_0009_0020 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0009_0020.pt", map_location=device
)
model_9.load_state_dict(ckpt_0009_0020["model_state_dict"])

ckpt_0010_0140 = torch.load(
    "data/DDPM/checkpoint/ddpm_checkpoint_0010_0140.pt", map_location=device
)
model_10.load_state_dict(ckpt_0010_0140["model_state_dict"])

# Demonstration of custom degradation scheme.
degradation_demo(
    model_9, device, [100, 75, 50, 25, 0], "", "custom_degradation.png"
)

# Take samples of generated images

SAMPLE_SIZE = 200

print(f"Sampling {SAMPLE_SIZE} images from each model.\n")

model_1.eval()
with torch.no_grad():
    plot_image_diffusion(
        model_1,
        [1.0, 0.5, 0.1, 0.05, 0.0],
        10,
        (1, 28, 28),
        device,
        "",
        "diffusion_plot_1.png",
    )
    _, sample_1 = compute_image_diffusion(
        model_1, [0.0], SAMPLE_SIZE, (1, 28, 28), device
    )

model_2.eval()
with torch.no_grad():
    plot_image_diffusion(
        model_2,
        [1.0, 0.5, 0.1, 0.05, 0.0],
        10,
        (1, 28, 28),
        device,
        "",
        "diffusion_plot_2.png",
    )
    _, sample_2 = compute_image_diffusion(
        model_2, [0.0], SAMPLE_SIZE, (1, 28, 28), device
    )

model_9.eval()
with torch.no_grad():
    plot_image_diffusion(
        model_9,
        [1.0, 0.5, 0.1, 0.05, 0.0],
        10,
        (1, 28, 28),
        device,
        "",
        "diffusion_plot_9.png",
    )
    _, sample_9, sample_9_direct = compute_image_diffusion_custom(
        model_9, [0.0], SAMPLE_SIZE, (1, 28, 28), device, direct=True
    )

model_10.eval()
with torch.no_grad():
    plot_image_diffusion(
        model_10,
        [1.0, 0.5, 0.2, 0.1, 0.0],
        10,
        (1, 28, 28),
        device,
        "",
        "diffusion_plot_10.png",
    )
    _, sample_10, sample_10_direct = compute_image_diffusion_custom(
        model_10, [0.0], SAMPLE_SIZE, (1, 28, 28), device, direct=True
    )

# Compute variances.
total_var_1 = compute_variance(sample_1[0])
print(f"Total variance of model 1 is {total_var_1}\n")

total_var_2 = compute_variance(sample_2[0])
print(f"Total variance of model 2 is {total_var_2}\n")

total_var_9 = compute_variance(sample_9[0])
print(f"Total variance of model 9 is {total_var_9}\n")

total_var_10 = compute_variance(sample_10[0])
print(f"Total variance of model 10 is {total_var_10}")

# Plot MNIST t-SNE plot in 2D, with classes.
plot_mnist_tsne("tsne_mnist.png")

density_gt, samples_fitted, samples_densities = compute_tsne_kl_div(
    [sample_1[0], sample_2[0], sample_9[0], sample_10[0]],
    ["Sample 1", "Sample 2", "Sample 9", "Sample 10"],
)

plot_sample_tsne(
    density_gt,
    samples_fitted[0],
    samples_fitted[1],
    samples_densities[0],
    samples_densities[1],
    "Sample of Model 1",
    "Sample of Model 2",
    "tsne_gaussian_models.png",
)
plot_sample_tsne(
    density_gt,
    samples_fitted[2],
    samples_fitted[3],
    samples_densities[2],
    samples_densities[3],
    "Sample of Model 9",
    "Sample of Model 10",
    "tsne_custom_models.png",
)

# Plot samples for all models
plot_samples(sample_1[0], 6, 6, "samples_1.png")
plot_samples(sample_2[0], 6, 6, "samples_2.png")
plot_samples(sample_9[0], 6, 6, "samples_9.png")
plot_samples(sample_10[0], 6, 6, "samples_10.png")

# Plot direct reconstructions for custom models.
print("Model 9, direct vs diffusion generation:")
plot_gt_direct_diffusion(
    sample_9_direct, sample_9[0], 10, "gt_direct_diffusion_9.png"
)

print("\n\nModel 10, direct vs diffusion generation:")
plot_gt_direct_diffusion(
    sample_10_direct, sample_10[0], 10, "gt_direct_diffusion_10.png"
)
