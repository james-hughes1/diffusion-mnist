import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from scipy.special import rel_entr
from typing import List
from tqdm import tqdm

from diffusiontools.models import DDPM, DMCustom


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


def compute_image_diffusion(
    model: DDPM, image_fractions: List[int], n_sample: int, size, device
):
    image_idx = [
        int(fraction * (model.n_T - 1)) for fraction in image_fractions
    ]
    images = []
    # Run model.sample method but with periodic saving to images.
    _one = torch.ones(n_sample, device=device)
    z_t = torch.randn(n_sample, *size, device=device)
    for i in tqdm(range(model.n_T, 0, -1)):
        alpha_t = model.alpha_t[i]
        beta_t = model.beta_t[i]

        # Save intermediate image generation.
        if i in image_idx:
            images.append(z_t.clone().detach().cpu().numpy())

        # First line of loop:
        z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * model.gt(
            z_t, (i / model.n_T) * _one
        )
        z_t /= torch.sqrt(1 - beta_t)

        if i > 1:
            # Last line of loop:
            z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
        # (We don't add noise at the final step - i.e., the last line of
        # the algorithm)
    images.append(z_t.clone().detach().cpu().numpy())

    return image_idx, images


def compute_image_diffusion_custom(
    model: DMCustom,
    image_fractions: List[int],
    n_sample: int,
    size,
    device,
    direct: bool = False,
) -> torch.Tensor:
    # Algorithm 2 from Bansal et al. Cold Diffusion paper.
    # Create random noise
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    dataset_mnist = MNIST("./data", train=False, download=True, transform=tf)
    z_t = torch.zeros((n_sample, 1, 28, 28), device=device)
    for i in range(n_sample):
        z_t[i, 0, :, :] = dataset_mnist.__getitem__(i)[0].to(device)

        # Maximal degradation.
        z_t[i, :, :, :] = model.degrade(
            z_t[i, :, :, :].unsqueeze(0), model.n_T - 1, device
        )

    image_idx = [int(fraction * (model.n_T)) for fraction in image_fractions]
    images = []

    _one = torch.ones(n_sample, device=device)
    direct_img = None
    for t in tqdm(range(model.n_T, 0, -1)):
        # Save intermediate image generation.
        if t in image_idx:
            images.append(z_t.clone().detach().cpu().numpy())

        # Reconstruction
        x_pred = model.gt(z_t, (t / model.n_T) * _one)

        if direct and t == model.n_T:
            direct_img = x_pred.clone().detach().cpu().numpy()

        # Degradation
        if t > 1:
            for i in range(n_sample):
                z_t[i, :, :, :] = model.degrade(
                    x_pred[i, :, :, :].unsqueeze(0), t - 1, device=device
                )
    if 0 in image_fractions:
        images.append(x_pred.clone().detach().cpu().numpy())

    return image_idx, images, direct_img


def plot_image_diffusion(
    model,
    image_fractions: List[int],
    n_sample: int,
    size,
    device,
    title: str,
    filename: str,
):
    n_image = len(image_fractions)
    if isinstance(model, DDPM):
        image_idx, images = compute_image_diffusion(
            model, image_fractions, n_sample, size, device
        )
    else:
        image_idx, images, _ = compute_image_diffusion_custom(
            model, image_fractions, n_sample, size, device
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


def plot_samples(samples: np.ndarray, n_row: int, n_col: int, filename: str):
    n_sample = n_row * n_col
    fig, ax = plt.subplots(n_row, n_col, figsize=(n_col * 2.5, n_row * 2.5))
    for i in range(n_sample):
        ax[i // n_col, i % n_col].imshow(samples[i, 0, :, :], cmap="grey")
        ax[i // n_col, i % n_col].set_xticks([])
        ax[i // n_col, i % n_col].set_yticks([])
    plt.tight_layout()
    plt.savefig("data/analysis/" + filename)


def plot_gt_direct_diffusion(
    direct_sample: np.ndarray,
    diffusion_sample: np.ndarray,
    n_plot: int,
    filename: str,
):
    n_sample = direct_sample.shape[0]
    # Load MNIST Dataset
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    dataset_mnist = MNIST("./data", train=False, download=True, transform=tf)
    gt_sample = np.zeros((n_sample, 1, 28, 28))
    gt_labels = np.zeros(n_sample)
    for i in range(n_sample):
        gt_sample[i, 0, :, :] = dataset_mnist.__getitem__(i)[0]
        gt_labels[i] = dataset_mnist.__getitem__(i)[1]

    fig, ax = plt.subplots(n_plot, 3, figsize=(7.5, n_plot * 2.5))
    for i in range(n_plot):
        # Plot last {n_plot} images and their reconstructions.
        ax[i, 0].imshow(gt_sample[-i, 0, :, :], cmap="grey")
        ax[i, 1].imshow(direct_sample[-i, 0, :, :], cmap="grey")
        ax[i, 2].imshow(diffusion_sample[-i, 0, :, :], cmap="grey")

        # Remove axis ticks.
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
    ax[0, 0].set(title="Ground Truth")
    ax[0, 1].set(title="Direct")
    ax[0, 2].set(title="Diffusion")

    plt.tight_layout()
    plt.savefig("data/analysis/" + filename)

    direct_MSE = np.zeros(n_sample)
    diffusion_MSE = np.zeros(n_sample)
    for i in range(n_sample):
        direct_MSE[i] = np.sqrt(
            np.mean((direct_sample[i] - gt_sample[i]) ** 2)
        )
        diffusion_MSE[i] = np.sqrt(
            np.mean((diffusion_sample[i] - gt_sample[i]) ** 2)
        )
    print(
        "Overall Average MSE for reconstructions"
        f" ... direct: {np.mean(direct_MSE):.5g}"
        f" diffusion: {np.mean(diffusion_MSE):.5g}"
    )
    for k in range(10):
        direct_MSE_k = np.mean(direct_MSE[np.where(gt_labels == k)])
        diffusion_MSE_k = np.mean(diffusion_MSE[np.where(gt_labels == k)])
        print(
            f"Digit {k} ... direct: {direct_MSE_k:.5g}"
            f" diffusion: {diffusion_MSE_k:.5g}"
        )


def compute_variance(sample: np.ndarray):
    pixel_var = np.var(sample, axis=0, ddof=1)
    total_var = np.sum(pixel_var)
    return total_var


def degradation_demo(
    model: DMCustom, device, t_list: List[int], title: str, filename: str
):
    # Get MNIST examples.
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    dataset_mnist = MNIST("./data", train=False, download=True, transform=tf)
    x = torch.zeros((4, 1, 28, 28), device=device)
    # The first instances of 1, 2, 3, 4 in MNIST (test) occur at these indices.
    for i, digit in enumerate([2, 1, 18, 4]):
        x[i, 0, :, :] = dataset_mnist.__getitem__(digit)[0].to(device)

    n_images = len(t_list)
    fig, ax = plt.subplots(4, n_images, figsize=(1.5 * n_images, 6))
    for j, t in enumerate(t_list):
        for i in range(4):
            z_t = model.degrade(x[i, 0, :, :].repeat(4, 1, 1, 1), t, device)
            ax[i, j].imshow(z_t[i, 0, :, :], cmap="grey")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 3:
                ax[i, j].set(xlabel=t_list[j])
    fig.supxlabel("Diffusion Timestep, t")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig("data/analysis/" + filename)


def compute_tsne_kl_div(samples: List[np.ndarray], labels: List[str]):
    # Load Test MNIST Data with pre-processing.
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    dataset_mnist = MNIST("./data", train=False, download=True, transform=tf)
    mnist_array = np.zeros((10000, 1, 28, 28))
    for i in range(10000):
        mnist_array[i, 0, :, :] = (
            dataset_mnist.__getitem__(i)[0].detach().cpu().numpy()
        )

    # Combine samples into single array.
    sample_sizes = np.cumsum([sample.shape[0] for sample in samples])
    full_dataset = np.concatenate([mnist_array, *samples], axis=0)
    full_dataset = full_dataset.reshape((full_dataset.shape[0], -1))

    # Fit TSNE model
    tsne = TSNE(n_components=2)
    full_dataset_fitted = tsne.fit_transform(full_dataset)

    # Make grid for plotting
    xx, yy = np.meshgrid(
        np.linspace(-100, 100, 201), np.linspace(-100, 100, 201)
    )
    XY = np.vstack([xx.flatten(), yy.flatten()]).T

    # Split dataset and fit GMM models.
    gt_fitted = full_dataset_fitted[:10000, :]
    model_GMM_gt = GaussianMixture(n_components=10)
    model_GMM_gt.fit(gt_fitted)
    loglh_gt = model_GMM_gt.score_samples(XY)
    density_gt = np.exp(loglh_gt).reshape(xx.shape)

    samples_fitted = []
    samples_densities = []
    for i in range(len(sample_sizes)):
        if i == 0:
            sample_i_fitted = full_dataset_fitted[
                10000 : 10000 + sample_sizes[0]
            ]
        else:
            sample_i_fitted = full_dataset_fitted[
                10000 + sample_sizes[i - 1] : 10000 + sample_sizes[i]
            ]
        samples_fitted.append(sample_i_fitted)

        # Fit GMM density.
        model_GMM_i = GaussianMixture(n_components=10)
        model_GMM_i.fit(sample_i_fitted)
        loglh_i = model_GMM_i.score_samples(XY)
        density_i = np.exp(loglh_i).reshape(xx.shape)
        samples_densities.append(density_i)

        # Estimate KL divergence.
        kl_div_estimate = np.sum(rel_entr(density_i, density_gt))
        print(
            f"KL Divergence between distribution of MNIST test dataset"
            f" and {labels[i]} is: {kl_div_estimate:.5g}"
        )

    return density_gt, samples_fitted, samples_densities


def plot_sample_tsne(
    density_gt: np.ndarray,
    sample_1_fitted: np.ndarray,
    sample_2_fitted: np.ndarray,
    density_1: np.ndarray,
    density_2: np.ndarray,
    label_1: str,
    label_2: str,
    filename: str,
):
    # Make grid for plotting
    xx, yy = np.meshgrid(
        np.linspace(-100, 100, 201), np.linspace(-100, 100, 201)
    )

    # Plot contour of Ground Truth density with sample scatters.
    fig, ax = plt.subplots(figsize=(10, 10))
    contour_plot = ax.contourf(
        xx, yy, density_gt, levels=10, cmap="GnBu", alpha=0.7, antialiased=True
    )
    cbar = plt.colorbar(contour_plot, format="%.0e")
    cbar.ax.set_ylabel("Test Dataset Density")
    ax.scatter(
        sample_1_fitted[:, 0], sample_1_fitted[:, 1], marker="+", c="black"
    )
    ax.scatter(
        sample_2_fitted[:, 0], sample_2_fitted[:, 1], marker=".", c="black"
    )

    # Add labels, legend.
    sample_1_marker = mlines.Line2D(
        [],
        [],
        c="black",
        marker="+",
        linestyle="None",
        markersize=10,
        label=label_1,
    )
    sample_2_marker = mlines.Line2D(
        [],
        [],
        c="black",
        marker=".",
        linestyle="None",
        markersize=10,
        label=label_2,
    )
    ax.legend(handles=[sample_1_marker, sample_2_marker])
    ax.set(xlabel="Embedding Dimension 1", ylabel="Embedding Dimension 2")
    plt.savefig("data/analysis/" + filename)


def plot_mnist_tsne(filename: str):
    # Load Test MNIST Data with pre-processing.
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    dataset_mnist = MNIST("./data", train=False, download=True, transform=tf)
    X = np.zeros((10000, 28 * 28))
    y = np.zeros(10000)
    for i in range(10000):
        X[i, :] = (
            dataset_mnist.__getitem__(i)[0].detach().cpu().numpy().flatten()
        )
        y[i] = dataset_mnist.__getitem__(i)[1]

    # Perform dimensionality reduction.
    tsne = TSNE(n_components=2)
    X_fitted = tsne.fit_transform(X)

    # Plot three samples
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter1 = ax.scatter(
        X_fitted[:, 0],
        X_fitted[:, 1],
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
    ax.set(xlabel="Embedding Dimension 1", ylabel="Embedding Dimension 2")
    plt.savefig("data/analysis/" + filename)
