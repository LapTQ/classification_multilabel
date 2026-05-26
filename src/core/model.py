import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)


class MultiLabelClassifyModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        class_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained efficientnet v2 s
        self.model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        # Loss function for multi-label classification
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Metrics for training, validation, testing
        self.train_f1_macro = MultilabelF1Score(
            num_labels=num_classes, average="macro"
        )
        self.val_f1_macro = MultilabelF1Score(
            num_labels=num_classes, average="macro"
        )
        self.test_f1_macro = MultilabelF1Score(
            num_labels=num_classes, average="macro"
        )

        # Per-class metrics for test evaluation
        self.test_f1_per_class = MultilabelF1Score(
            num_labels=num_classes, average=None
        )
        self.test_precision_per_class = MultilabelPrecision(
            num_labels=num_classes, average=None
        )
        self.test_recall_per_class = MultilabelRecall(
            num_labels=num_classes, average=None
        )

        self.class_names: Optional[List[str]] = class_names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        f1 = self.train_f1_macro(logits, y.long())
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        self.log(
            "train_f1_macro",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_f1_macro(logits, y.long())

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        self.log(
            "val_f1_macro",
            self.val_f1_macro,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], batch_idx: int
    ) -> None:
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.test_f1_macro(logits, y.long())
        self.test_f1_per_class(logits, y.long())
        self.test_precision_per_class(logits, y.long())
        self.test_recall_per_class(logits, y.long())

        self.log("test_loss", loss, on_epoch=True, batch_size=x.shape[0])
        self.log(
            "test_f1_macro",
            self.test_f1_macro,
            on_epoch=True,
            batch_size=x.shape[0],
        )

    def on_test_epoch_end(self) -> None:
        f1_vals = self.test_f1_per_class.compute().cpu().numpy()
        prec_vals = self.test_precision_per_class.compute().cpu().numpy()
        rec_vals = self.test_recall_per_class.compute().cpu().numpy()

        self.test_f1_per_class.reset()
        self.test_precision_per_class.reset()
        self.test_recall_per_class.reset()

        classes = (
            self.class_names
            if self.class_names
            else [f"Class_{i}" for i in range(len(f1_vals))]
        )

        # Plot bar chart of per-class metrics
        plt.figure(figsize=(14, 8))
        x = np.arange(len(classes))
        width = 0.25

        plt.bar(x - width, f1_vals, width, label="F1 Score", color="skyblue")
        plt.bar(x, prec_vals, width, label="Precision", color="salmon")
        plt.bar(x + width, rec_vals, width, label="Recall", color="lightgreen")

        plt.xlabel("Classes", fontsize=12)
        plt.ylabel("Scores", fontsize=12)
        plt.title("Per-class Metrics on Test Dataset", fontsize=14)
        plt.xticks(x, classes, rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.legend(loc="upper right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        save_path = os.path.join(
            self.trainer.default_root_dir, "test_class_metrics.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        print(f"\n[INFO] Per-class metrics chart saved to: {save_path}")
        plt.close()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_epochs = 3
        # Ensure trainer epochs is parsed
        max_epochs = (
            self.trainer.max_epochs
            if self.trainer.max_epochs is not None and self.trainer.max_epochs > 0
            else 100
        )
        main_epochs = max_epochs - warmup_epochs

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=self.hparams.lr * 0.01
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
