import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from torchvision.utils import save_image, make_grid


def train_ddpm(
    ddpm: nn.Module,
    optim: torch.optim.Optimizer,
    dataloader: DataLoader,
    accelerator: Accelerator,
    n_epoch: int,
    save_interval: int,
    sample_path: str,
    model_path: str,
    config_id: str,
):
    losses = []
    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
        for x, _ in pbar:
            optim.zero_grad()

            loss = ddpm(x)

            loss.backward()
            # ^Technically should be `accelerator.backward(loss)` but not
            # necessary for local training

            losses.append(loss.item())
            avg_loss = np.average(losses[min(len(losses) - 100, 0) :])
            # Show running average of loss in progress bar
            pbar.set_description(f"loss: {avg_loss:.3g}")

            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), accelerator.device)
            grid = make_grid(xh, nrow=4)

            # Save samples to `./contents` directory
            sample_filename = (
                sample_path + f"ddpm_sample_{config_id:04d}_{i:04d}.png"
            )
            save_image(grid, sample_filename)

            # save model
            model_filename = (
                model_path + f"ddpm_model_{config_id:04d}_{i:04d}.pth"
            )
            if i % save_interval == 0:
                torch.save(ddpm.state_dict(), model_filename)
