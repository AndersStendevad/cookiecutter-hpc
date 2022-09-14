"""Trainer
=======
Trainer class for training the model.
"""

import torch.nn as nn
import torch
import torch.optim as optim

from abc import ABC, abstractmethod

torch.backends.cudnn.enabled = False


class Train_base(ABC):
    """Base class for training the model."""

    def __init__(
        self,
        model: nn.Module,
        optim: optim.Optimizer,
        criterion: nn.Module,
        clip: float,
        device,
    ):
        """Initialize the trainer."""

        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.clip = clip
        self.device = device

    @abstractmethod
    def train(self, iterator: torch.utils.data.DataLoader, progress):
        pass

    @abstractmethod
    def eval(self, iterator: torch.utils.data.DataLoader):
        pass


class Trainer(Train_base):
    """
    Trainer class for training the model.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: optim.Optimizer,
        criterion: nn.Module,
        clip: float,
        device,
    ):
        """
        Initialize the trainer.

        Parameters
        ----------
        model : nn.Module
            The model to train.
        optim : optim.Optimizer
            The optimizer to use.
        criterion : nn.Module
            The loss function to use.
        clip : float
            The gradient clipping value.
        device : torch.device
            The device to use.

        Returns
        -------
        Trainer : Trainer
        """

        super().__init__(model, optim, criterion, clip, device)

    def train(self, iterator: torch.utils.data.DataLoader, progress=None):
        """
        Train the model.

        Parameters
        ----------
        iterator : torch.utils.data.DataLoader
            The iterator to use.
        progress : pkbar.ProgressBar
            The progress bar to use.

        Returns
        -------
        epoch_loss : float
            The loss of the epoch.
        """
        self.model.train()

        epoch_loss = 0
        loss_count = 0

        for i, seq in enumerate(iterator):

            seq = seq.to(self.device).long()

            self.optim.zero_grad()

            inputs = {"input_ids": seq}

            output = self.model(**inputs).logits

            output = output.view(-1, output.shape[-1])
            seq_view = seq.reshape(-1)

            loss = self.criterion(output, seq_view)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            self.optim.step()

            epoch_loss += loss.detach()
            loss_count += seq.shape[0]

            if progress:
                progress.update(
                    i,
                    values=[
                        ("Loss", epoch_loss),
                        ("Avg loss", epoch_loss / loss_count),
                        (
                            "Cached GB",
                            round(torch.cuda.memory_reserved(0) / 1024**3, 1),
                        ),
                    ],
                )

            del seq
            del seq_view
            del output
            del loss

        if progress:
            progress.add(1)

        return epoch_loss / loss_count

    def eval(self, iterator: torch.utils.data.DataLoader):
        """
        Evaluate the model.

        Parameters
        ----------
        iterator : torch.utils.data.DataLoader
            The iterator to use.

        Returns
        -------
        epoch_loss : float
            The loss of the epoch.
        """
        self.model.eval()

        loss_count = 0
        epoch_loss = 0

        with torch.no_grad():

            for _, seq in enumerate(iterator):
                seq = seq.to(self.device).long()

                output = self.model(seq).logits

                output = output.view(-1, output.shape[-1])
                seq_view = seq.reshape(-1)

                loss = self.criterion(output, seq_view)
                loss_count += seq.shape[1]

                epoch_loss += float(loss.item())

        return epoch_loss / loss_count
