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
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    accelerator: Accelerator,
    n_epoch: int,
    save_interval: int,
    sample_path: str,
    checkpoint_path: str,
    config_id: str,
):
    losses_train_batch = []
    losses_train_epoch = []
    losses_val_epoch = []

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(
            dataloader_train
        )  # Wrap our loop with a visual progress bar
        for x, _ in pbar:
            optim.zero_grad()

            loss = ddpm(x)

            accelerator.backward(loss)
            # ^Technically should be `accelerator.backward(loss)` but not
            # necessary for local training

            losses_train_batch.append(loss.item())
            avg_loss_train = np.average(
                losses_train_batch[min(len(losses_train_batch) - 100, 0) :]
            )
            # Show running average of loss in progress bar
            pbar.set_description(
                f"Epoch {i:04d}, train loss: {avg_loss_train:.3g}"
            )

            optim.step()

        ddpm.eval()
        with torch.no_grad():
            # Compute validation loss
            losses_val_batch = []
            for x, _ in dataloader_val:
                loss = ddpm(x)
                losses_val_batch.append(loss.item())

            # Save losses for epoch
            losses_val_epoch.append(np.average(losses_val_batch))
            losses_train_epoch.append(avg_loss_train)
            print(
                f"Epoch {i:04d} had train loss: {avg_loss_train:.3g}"
                + f" val loss: {losses_val_epoch[-1]:.3g}"
            )

            # Save samples
            xh = ddpm.sample(16, (1, 28, 28), accelerator.device)
            grid = make_grid(xh, nrow=4)
            sample_filename = (
                sample_path + f"ddpm_sample_{config_id:04d}_{i:04d}.png"
            )
            save_image(grid, sample_filename)

            # Checkpoint
            if i % save_interval == 0:
                checkpoint_filename = (
                    checkpoint_path
                    + f"ddpm_checkpoint_{config_id:04d}_{i:04d}.pt"
                )
                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": ddpm.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "losses_train": losses_train_epoch,
                        "losses_val": losses_val_epoch,
                    },
                    checkpoint_filename,
                )
