import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class DiffusionProcess(ABC):
    """Abstract base class for all continuous and discrete diffusion processes."""

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the training loss and the predicted clean state (x_0)."""
        pass

    @abstractmethod
    def ddpm_step(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, batch_vec: torch.Tensor, edge_index: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Performs a single Ancestral/DDPM sampling step."""
        pass

    @abstractmethod
    def ddim_step(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, batch_vec: torch.Tensor, edge_index: torch.Tensor, field: torch.Tensor, eta: float) -> torch.Tensor:
        """Performs a single non-Markovian DDIM sampling step."""
        pass
    
    @abstractmethod
    def d3pm_step(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, batch_vec: torch.Tensor, edge_index: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Performs a single Discrete Denoising (D3PM) sampling step."""
        pass