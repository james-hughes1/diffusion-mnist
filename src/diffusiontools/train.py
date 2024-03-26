"""!@file train.py
    @brief Module containing procedures used to train and save diffusion
    models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm


def train_model(
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
    """!@brief Handles the training process for a diffusion model.
    @param ddpm Diffusion model object; either DDPM or DMCustom class
    instance
    @param dataloader_train Iterable that yields pairs of matched ground
    truth and degraded image samples for training
    @param dataloader_val Iterable that yields pairs of matched ground
    truth and degraded image samples for model validation
    @param accelerator accelerate package object which simplifies the
    management of CPU/GPU processing devices
    @param n_epoch Total number of epochs to train for before stopping
    @param save_interval Number of epochs to wait for between saving models
    @param sample_path Specifies where to save model samples (deprecated)
    @param checkpoint_path Specifies where to save model checkpoint files
    @param config_id Specifies configuration id to use to label model
    checkpoint files
    """
    losses_train_epoch = []
    losses_val_epoch = []

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(
            dataloader_train
        )  # Wrap our loop with a visual progress bar
        losses_train_batch = []
        for x, _ in pbar:
            optim.zero_grad()

            loss = ddpm(x)

            accelerator.backward(loss)
            # ^Technically should be `accelerator.backward(loss)` but not
            # necessary for local training

            losses_train_batch.append(loss.item())

            # Rolling average loss for display.
            avg_loss_train = np.average(losses_train_batch)
            # Show rolling average of loss in progress bar.
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

            # Checkpoint
            if i % save_interval == 0:
                # Save model.
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
