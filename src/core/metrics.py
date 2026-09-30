from typing import Optional

import torch
from torchmetrics import Metric


class _MaskedMultilabelMetricBase(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor

    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[str] = "macro",
    ) -> None:
        super().__init__()
        self.num_labels: int = num_labels
        self.threshold: float = threshold
        self.average: Optional[str] = average

        self.add_state(
            "tp",
            default=torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> None:
        preds = (torch.sigmoid(logits) >= self.threshold).float()
        valid = mask.bool()
        targets = targets.float()

        tp = ((preds == 1.0) & (targets == 1.0) & valid).sum(dim=0).float()
        fp = ((preds == 1.0) & (targets == 0.0) & valid).sum(dim=0).float()
        fn = ((preds == 0.0) & (targets == 1.0) & valid).sum(dim=0).float()

        self.tp += tp
        self.fp += fp
        self.fn += fn


class MaskedMultilabelF1Score(_MaskedMultilabelMetricBase):
    def compute(self) -> torch.Tensor:
        precision = self.tp / (self.tp + self.fp).clamp(min=1e-7)
        recall = self.tp / (self.tp + self.fn).clamp(min=1e-7)
        f1_per_class = 2 * (precision * recall) / (precision + recall).clamp(min=1e-7)

        if self.average is None:
            return f1_per_class
        elif self.average == "macro":
            has_samples = (self.tp + self.fp + self.fn) > 0
            if has_samples.sum() == 0:
                return torch.tensor(0.0, device=self.tp.device)
            return f1_per_class[has_samples].mean()
        else:
            raise ValueError(f"Unsupported average: {self.average}")


class MaskedMultilabelPrecision(_MaskedMultilabelMetricBase):
    def compute(self) -> torch.Tensor:
        precision = self.tp / (self.tp + self.fp).clamp(min=1e-7)

        if self.average is None:
            return precision
        elif self.average == "macro":
            has_samples = (self.tp + self.fp + self.fn) > 0
            if has_samples.sum() == 0:
                return torch.tensor(0.0, device=self.tp.device)
            return precision[has_samples].mean()
        else:
            raise ValueError(f"Unsupported average: {self.average}")


class MaskedMultilabelRecall(_MaskedMultilabelMetricBase):
    def compute(self) -> torch.Tensor:
        recall = self.tp / (self.tp + self.fn).clamp(min=1e-7)

        if self.average is None:
            return recall
        elif self.average == "macro":
            has_samples = (self.tp + self.fp + self.fn) > 0
            if has_samples.sum() == 0:
                return torch.tensor(0.0, device=self.tp.device)
            return recall[has_samples].mean()
        else:
            raise ValueError(f"Unsupported average: {self.average}")
