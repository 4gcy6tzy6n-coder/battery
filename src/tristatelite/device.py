"""Compute-device selection helpers."""

import torch


def select_device() -> torch.device:
    """Select the best available device in the project-defined order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
