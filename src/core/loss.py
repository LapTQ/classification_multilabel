import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        unreduced_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        masked_loss = unreduced_loss * mask
        return masked_loss.sum() / mask.sum().clamp(min=1.0)
