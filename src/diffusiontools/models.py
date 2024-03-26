"""!@file models.py
    @brief Module containing classes and procedures used to build diffusion
    models.
"""


from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def ddpm_schedules(
    beta1: float, beta2: float, T: int
) -> Dict[str, torch.Tensor]:
    """!@brief Returns pre-computed schedules for DDPM sampling with a linear
    noise schedule.
    @param beta1 Minimal noise parameter
    @param beta2 Taximal noise parameter
    @param T Total number of discrete diffusion time steps
    @returns Dict[str,torch.Tensor] Dictionary storing alpha and beta noise
    parameter vectors
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(
        0, T + 1, dtype=torch.float32
    ) / T + beta1
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))

    return {"beta_t": beta_t, "alpha_t": alpha_t}


class CNNBlock(nn.Module):
    """!@brief Class instantiating a convolutional block with layer
    normalisation and non-linear activation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        """!@brief Initialisation for CNNBlock class
        @param in_channels Number of input channels
        @param out_channels Number of output channels
        @param expected_shape Image shape (height, width)
        @param act Non-linear activation function
        @param kernel_size Size of convolutional kernel to use
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.LayerNorm(expected_shape),
            act(),
        )

    def forward(self, x):
        """!@brief Forward pass function for CNNBlock.
        @param x Input from previous hidden layer
        @returns torch.tensor Output to pass to next hidden layer
        """
        return self.net(x)


class CNN(nn.Module):
    """!@brief Defines CNN class used to reconstruct images during
    diffusion.
    """

    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        """!@brief Initialisation for CNN class.
        @param in_channels Number of input channels
        @param expected_shape image shape as (height, width)
        @param n_hidden Tuple containing number of hidden channels per
        hidden layer
        @param kernel_size Convolution kernel size
        @param last_kernel_size Used to specify different kernel size in
        final layer
        @param time_embedding Determines dimensionality of temporal
        embeddings for diffusion (which is later doubled before combining
        with the network)
        @param act Non-linear activation function to use in the hidden
        neurons
        """
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        # This part is literally just to put the single scalar "t" into the CNN
        # in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """!@brief Produces embedding vectors for the time steps passed during
        the input to the network.
        @param t Tensor of time steps, one for each batch member
        @returns torch.Tensor Embedded vectors that have been passed
        through trigonometric functions and then learned linear network
        """
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """!@brief Defines the forward pass for the model.
        @param x Image stack with shape (batch, chan, height, width)
        @param t Time step stack with shape (batch,)
        @returns embed Final output of the model
        """
        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed


class DDPM(nn.Module):
    """!@brief Defines Gaussian diffusion model class."""

    def __init__(
        self,
        gt,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        """!@brief Initialisation for DDPM class.
        @param gt Trainable model used to approximate the reconstruction
        process
        @param betas Determines the extremes of the noise levels along the
        diffusion process
        @param n_T Number of discrete diffusion time steps
        @param criterion Loss function to use for the denoising process
        """
        super().__init__()

        self.gt = gt

        noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!@brief Forward pass for the DDPM model.
        @details Implements Algorithm 18.1 in Understanding Deep Learning,
        found at http://udlbook.com. Note that unusually, this forward
        method returns the loss function of the predicted error, rather
        than the error prediction itself.
        @param x Batched input
        @returns torch.Tensor Loss function evaluated on the predicted
        error compared to the true error
        """

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[
            t, None, None, None
        ]  # Get right shape for broadcasting

        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t. Loss is what we
        # return.

        return self.criterion(eps, self.gt(z_t, t / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """!@brief Reverse diffusion sampling for the DDPM model.
        @details Implements Algorithm 18.2 in Understanding Deep Learning,
        found at http://udlbook.com.
        @param n_sample Number of images to generate
        @param size Image size tuple in the format (channel, height, width)
        @returns z_t Stack of generated image samples
        """

        _one = torch.ones(n_sample, device=device)
        z_t = torch.randn(n_sample, *size, device=device)
        for i in range(self.n_T, 0, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(
                z_t, (i / self.n_T) * _one
            )
            z_t /= torch.sqrt(1 - beta_t)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of
            # the algorithm)

        return z_t


class DMCustom(nn.Module):
    """!@brief Class defining custom diffusion model."""

    def __init__(
        self,
        gt,
        alphas: Tuple[float, float],
        n_T: int,
        size: Tuple[int, int],
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        """!@brief Initialisation for DMCustom class.
        @param gt Trainable model used to approximate the reconstruction
        process
        @param alphas Determines the extremes of the noise levels along the
        diffusion process
        @param n_T Number of discrete diffusion time steps
        @param size Image size tuple in the format (channel, height, width)
        @param criterion Loss function to use for the denoising process
        """
        super().__init__()

        self.gt = gt

        noise_schedule = ddpm_schedules(alphas[0], alphas[1], n_T)

        # Note that alpha1 should be chosen such that there is no
        # degradation at the t=0 step.
        # Note also that alpha_t here is set with uniform
        self.register_buffer("noise_t", noise_schedule["beta_t"])
        self.noise_t

        self.n_T = n_T
        self.criterion = criterion
        self.size = size

    def degrade(self, x: torch.Tensor, t: int, device) -> torch.Tensor:
        """!@brief Implements custom degradation operation
        @param x Batched image tensor
        @param t Integer discrete time-step; note that the same
        degradation is applied to all images in the stack
        @param device torch.device object which enables GPU or CPU to be
        specified
        @returns z_t Stack of images with degradation applied
        """
        z_t = x.clone()
        batch_size = z_t.shape[0]
        size = self.size
        delta1 = (
            (torch.rand((size[1], size[2]), device=device) - 0.5)
            * 2
            * self.noise_t[t]
            * size[1]
        )
        delta2 = (
            (torch.rand((size[1], size[2]), device=device) - 0.5)
            * 2
            * self.noise_t[t]
            * size[2]
        )
        rows, cols = torch.meshgrid(
            torch.arange(size[1], device=device),
            torch.arange(size[2], device=device),
            indexing="ij",
        )
        cols.unsqueeze(0).repeat(batch_size, 1, 1)
        rows.unsqueeze(0).repeat(batch_size, 1, 1)
        cols = (cols + delta1.int()) % size[1]
        rows = (rows + delta2.int()) % size[2]
        for i in range(size[1]):
            for j in range(size[2]):
                z_t[:, 0, i, j], z_t[:, 0, rows[i, j], cols[i, j]] = (
                    z_t[:, 0, rows[i, j], cols[i, j]],
                    z_t[:, 0, i, j],
                )
        return z_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """!@brief Forward pass for the DMCustom model.
        @details Implements Algorithm 18.1 in Understanding Deep Learning,
        found at http://udlbook.com. Note that unusually, this forward
        method returns the loss function of the predicted error, rather
        than the error prediction itself.
        @param x Batched input
        @returns torch.Tensor Loss function evaluated on the predicted
        error compared to the true error
        """
        # Choose random time step and degrade image batch accordingly.
        t = torch.randint(1, self.n_T, (1,), device=x.device)
        z_t = x.clone()
        z_t = self.degrade(z_t, t, x.device)

        _one = torch.ones(x.shape[0], device=x.device)
        # Note we have changed from predicting error (eps) to x itself.
        return self.criterion(x, self.gt(z_t, (t / self.n_T) * _one))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """!@brief Reverse diffusion sampling for the DMCustom model.
        @details Implements Algorithm 18.2 in Understanding Deep Learning,
        found at http://udlbook.com.
        @param n_sample Number of images to generate
        @param size Image size tuple in the format (channel, height, width)
        @returns z_t Stack of generated image samples
        """
        # Create random noise
        pos = torch.arange(
            0, n_sample * size[0] * size[1] * size[2], device=device
        )
        pos = (pos % 5 == 0).reshape((n_sample, *size)) * 1
        random_vals = torch.rand(n_sample, *size, device=device)
        z_t = pos * (0.25 + random_vals / 4) + (pos - 1) * (
            0.49 + random_vals / 100
        )
        # Maximal degradation.
        z_t = self.degrade(z_t, self.n_T - 1, device)

        _one = torch.ones(n_sample, device=device)
        for t in range(self.n_T, 0, -1):
            # Reconstruction
            x_pred = self.gt(z_t, (t / self.n_T) * _one)

            # Degradation
            z_t = self.degrade(x_pred, t - 1, device)

        return z_t
