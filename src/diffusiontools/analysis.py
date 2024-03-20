import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
from typing import List
from tqdm import tqdm

from diffusiontools.models import DDPM


def plot_learning_curve(
    losses_train: List[float],
    losses_val: List[float],
    title: str,
    filename: str,
):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(losses_train, label="Training Loss", color="red")
    ax.plot(losses_val, label="Validation Loss", color="blue")
    ax.set(title=title, xlabel="Epoch", ylabel="Loss")
    ax.legend()
    plt.savefig("data/analysis/" + filename)
    print(f"Final training loss was {losses_train[-1]:.5g}")
    print(f"Final validation loss was {losses_val[-1]:.5g}")


def compute_image_diffusion(
    ddpm: DDPM, image_fractions: List[int], n_sample: int, size, device
):
    image_idx = [
        int(fraction * (ddpm.n_T - 1)) for fraction in image_fractions
    ]
    images = []

    # Run DDPM.sample method but with periodic saving to images.
    _one = torch.ones(n_sample, device=device)
    z_t = torch.randn(n_sample, *size, device=device)
    for i in tqdm(range(ddpm.n_T, 0, -1)):
        alpha_t = ddpm.alpha_t[i]
        beta_t = ddpm.beta_t[i]

        # First line of loop:
        z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * ddpm.gt(
            z_t, (i / ddpm.n_T) * _one
        )
        z_t /= torch.sqrt(1 - beta_t)

        if i > 1:
            # Last line of loop:
            z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
        # (We don't add noise at the final step - i.e., the last line of
        # the algorithm)

        # Save intermediate image generation.
        if i in image_idx:
            images.append(z_t.clone().detach().cpu().numpy())
    images.append(z_t.clone().detach().cpu().numpy())

    return image_idx, images


def plot_image_diffusion(
    ddpm: DDPM,
    image_fractions: List[int],
    n_sample: int,
    size,
    device,
    title: str,
    filename: str,
):
    n_image = len(image_fractions)
    image_idx, images = compute_image_diffusion(
        ddpm, image_fractions, n_sample, size, device
    )

    fig, ax = plt.subplots(
        n_sample, n_image, figsize=(1.5 * n_image, 1.5 * n_sample)
    )
    for i in range(n_sample):
        for j in range(n_image):
            ax[i, j].imshow(images[j][i, 0, :, :], cmap="grey")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == n_sample - 1:
                ax[i, j].set(xlabel=image_idx[j])
            if j == 0:
                ax[i, j].set(ylabel=i)
    fig.supxlabel("Diffusion Timestep, t")
    fig.supylabel("Sample Index")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("data/analysis/" + filename)


def compute_variance(sample: np.ndarray):
    pixel_var = np.var(sample, axis=0, ddof=1)
    total_var = np.sum(pixel_var)
    return total_var


def plot_tsne(sample1: np.ndarray, sample2: np.ndarray, filename: str):
    n1, n2 = len(sample1), len(sample2)

    # Load Test MNIST Data with pre-processing.
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    dataset_mnist = MNIST("./data", train=False, download=True, transform=tf)
    X = np.zeros((10000 + n1 + n2, 28 * 28))
    y = np.zeros(10000)
    for i in range(10000):
        X[i, :] = (
            dataset_mnist.__getitem__(i)[0].detach().cpu().numpy().flatten()
        )
        y[i] = dataset_mnist.__getitem__(i)[1]
    for i in range(n1):
        X[i + 10000, :] = sample1[i].flatten()
    for i in range(n2):
        X[i + 10000 + n1, :] = sample2[i].flatten()

    # Perform dimensionality reduction.
    tsne = TSNE(n_components=2)
    X_fitted = tsne.fit_transform(X)

    # Plot three samples
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter1 = ax.scatter(
        X_fitted[:10000, 0],
        X_fitted[:10000, 1],
        lw=0.01,
        c=y,
        cmap="gist_rainbow",
        marker=".",
        alpha=0.5,
    )
    legend1 = ax.legend(
        *scatter1.legend_elements(), loc="lower left", title="MNIST Classes"
    )
    ax.add_artist(legend1)
    ax.scatter(
        X_fitted[10000 : 10000 + n1, 0],
        X_fitted[10000 : 10000 + n1 :, 1],
        lw=0.05,
        c="black",
        marker="*",
    )
    ax.scatter(
        X_fitted[10000 + n1 :, 0],
        X_fitted[10000 + n1 :, 1],
        lw=0.05,
        c="black",
        marker="P",
    )
    sample1_marker = mlines.Line2D(
        [],
        [],
        c="black",
        marker="*",
        linestyle="None",
        markersize=10,
        label="Sample 1",
    )
    sample2_marker = mlines.Line2D(
        [],
        [],
        c="black",
        marker="P",
        linestyle="None",
        markersize=10,
        label="Sample 2",
    )
    ax.legend(handles=[sample1_marker, sample2_marker], loc="upper right")
    ax.set(
        title="t-SNE Embedding of MNIST Test Data",
        xlabel="Embedding Dimension 1",
        ylabel="Embedding Dimension 2",
    )
    plt.savefig("data/analysis/" + filename)
