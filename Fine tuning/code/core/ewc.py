# =============================================================================
# core/ewc.py
# Elastic Weight Consolidation (EWC) — Prevents Catastrophic Forgetting
# =============================================================================
#
# Theory:
#   After Stage 2a (printed line fine-tuning), we want Stage 2b (handwritten
#   adaptation) to NOT overwrite the weights that are critical for printed text.
#
#   EWC adds a penalty term to the loss:
#       L_total = L_ce + (lambda/2) * sum_i [ F_i * (theta_i - theta*_i)^2 ]
#
#   Where:
#       F_i      = Fisher Information for parameter i (how "important" it is
#                  for the task we want to preserve — printed OCR)
#       theta*_i = the optimal weights after Stage 2a (the anchor)
#       theta_i  = the current weights being updated in Stage 2b
#
#   Fisher Information is approximated as the expected squared gradient of
#   the log-likelihood: F_i ≈ E[ (d log p(y|x) / d theta_i)^2 ]
#   In practice we average this over a sample of the printed training data.
#
# =============================================================================

import logging
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EWC:
    """
    Elastic Weight Consolidation for TrOCR.

    Workflow:
        1. After Stage 2a completes, instantiate EWC with the trained model.
        2. Call `compute_fisher(dataloader)` to estimate Fisher Information
           using printed line data — this takes ~5-10 min on Colab.
        3. During Stage 2b, call `ewc_loss(model)` to get the penalty term.
        4. Add it to the cross-entropy loss: loss = ce_loss + ewc.ewc_loss(model)

    Args:
        model:      The TrOCR model AFTER Stage 2a training.
        lambda_ewc: Penalty weight. Higher = more protection of printed skill.
                    Typical range: 100 ~ 1000. Start with 500 and tune.
        device:     torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 500.0,
        device: torch.device = None,
    ):
        self.model      = model
        self.lambda_ewc = lambda_ewc
        self.device     = device or next(model.parameters()).device

        # theta* — a snapshot of weights at the end of Stage 2a
        # We store only parameters that require grad (trainable params)
        self._params_star: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._params_star[name] = param.data.clone().detach()

        # F — Fisher Information diagonal (same shape as each parameter)
        self._fisher: Dict[str, torch.Tensor] = {}
        self._fisher_computed = False

        logger.info(
            f"[EWC] Initialized with lambda={lambda_ewc} | "
            f"tracked params={len(self._params_star)}"
        )

    @torch.no_grad()
    def _zero_fisher(self):
        """Initialize Fisher accumulators to zero."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._fisher[name] = torch.zeros_like(param.data)

    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: int = 200,
        fp16: bool = False,
    ) -> None:
        """
        Estimate the diagonal Fisher Information Matrix over the printed dataset.
        """
        logger.info(
            f"[EWC] Computing Fisher Information over {num_samples} batches "
            f"of printed data..."
        )
        self.model.eval()
        self._zero_fisher()

        scaler = torch.amp.GradScaler("cuda", enabled=fp16)
        
        # ==========================================
        # BUG FIX 1: KHỞI TẠO DUMMY OPTIMIZER BÊN NGOÀI
        # Khởi tạo 1 lần duy nhất để tránh Python cấp phát lại trùng ID ô nhớ
        # ==========================================
        dummy_optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-8)
        
        batch_count = 0
        inf_skipped = 0

        for batch in tqdm(dataloader, total=num_samples, desc="Fisher"):
            if batch_count + inf_skipped >= num_samples:
                break

            pixel_values = batch["pixel_values"].to(self.device)
            labels       = batch["labels"].to(self.device)

            # Dùng luôn dummy_optimizer để xóa gradient cũ thay cho self.model.zero_grad()
            dummy_optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=fp16):
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss

            # Backward to get gradients
            if fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(dummy_optimizer)

                # ==========================================
                # FIX 3: SKIP BATCHES WITH INF GRADIENTS
                # When FP16 scale factor is too large, unscale_
                # produces inf gradients. In normal training the
                # optimizer skips these via scaler.step(). Here
                # we must skip manually to prevent inf in Fisher.
                # ==========================================
                found_inf = False
                for opt_state in dummy_optimizer.state_dict()["state"].values():
                    if "found_inf_per_device" in opt_state:
                        found_inf = True
                        break
                # Alternative check: inspect the scaler's internal flag
                if not found_inf:
                    # Check gradients directly for any inf/nan
                    for _, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            if not torch.isfinite(param.grad).all():
                                found_inf = True
                                break

                if found_inf:
                    inf_skipped += 1
                    dummy_optimizer.zero_grad()
                    scaler.update()
                    continue
            else:
                loss.backward()

            # Accumulate squared gradients into Fisher diagonal
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # ==========================================
                    # FIX: ÉP SANG FLOAT32 TRƯỚC KHI BÌNH PHƯƠNG
                    # ==========================================
                    grad_fp32 = param.grad.data.clone().to(torch.float32)
                    self._fisher[name] += grad_fp32.pow(2)

            batch_count += 1
            
            # ==========================================
            # FIX 2: RESET TRẠNG THÁI SCALER
            # Báo cho PyTorch biết vòng lặp đã xong để xả cờ unscale_
            # ==========================================
            if fp16:
                scaler.update()

        # Normalize by number of batches processed
        if batch_count > 0:
            for name in self._fisher:
                self._fisher[name] /= batch_count

        # ==========================================
        # FIX 4: CLAMP INF/NAN VALUES IN FISHER
        # Safety net — replace any remaining inf/nan with 0
        # (a param with inf Fisher would freeze that weight entirely)
        # ==========================================
        clamped_count = 0
        for name in self._fisher:
            mask = ~torch.isfinite(self._fisher[name])
            if mask.any():
                clamped_count += mask.sum().item()
                self._fisher[name] = torch.where(
                    mask,
                    torch.zeros_like(self._fisher[name]),
                    self._fisher[name],
                )
        if clamped_count > 0:
            logger.warning(
                f"[EWC] Clamped {clamped_count} inf/nan Fisher values to 0."
            )
        if inf_skipped > 0:
            logger.info(
                f"[EWC] Skipped {inf_skipped} batches with inf gradients "
                f"(FP16 overflow). Used {batch_count} clean batches."
            )

        # ==========================================
        # CLEANUP: XÓA 1.5GB GRADIENTS 
        # ==========================================
        self.model.zero_grad(set_to_none=True)
        del dummy_optimizer
        # ==========================================

        self._fisher_computed = True
        logger.info(
            f"[EWC] Fisher computation complete. "
            f"Batches used: {batch_count}. "
            f"Mean Fisher magnitude: "
            f"{sum(f.mean().item() for f in self._fisher.values()) / len(self._fisher):.6f}"
        )

    def ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty for the current model weights.

        penalty = (lambda/2) * sum_i [ F_i * (theta_i - theta*_i)^2 ]

        This is added to the base cross-entropy loss during Stage 2b to
        penalize large deviations from the Stage 2a weights, proportional
        to how important each weight was for the printed OCR task.

        Args:
            model: The model currently being trained in Stage 2b.

        Returns:
            Scalar tensor representing the EWC penalty.
        """
        if not self._fisher_computed:
            raise RuntimeError(
                "Fisher Information has not been computed yet. "
                "Call ewc.compute_fisher(dataloader) after Stage 2a."
            )

        penalty = torch.tensor(0.0, device=self.device)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._fisher or name not in self._params_star:
                continue

            fisher = self._fisher[name]
            theta_star = self._params_star[name]

            # Squared deviation from Stage 2a optimal, weighted by importance
            penalty = penalty + (fisher * (param - theta_star).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * penalty

    def save(self, path: str) -> None:
        """Persist Fisher and theta* to disk for checkpoint resume."""
        torch.save(
            {
                "fisher":      self._fisher,
                "params_star": self._params_star,
                "lambda_ewc":  self.lambda_ewc,
                "computed":    self._fisher_computed,
            },
            path,
        )
        logger.info(f"[EWC] State saved to {path}")

    def load(self, path: str) -> None:
        """Load persisted EWC state (for resume after Colab restart)."""
        state = torch.load(path, map_location=self.device)
        self._fisher          = {k: v.to(self.device) for k, v in state["fisher"].items()}
        self._params_star     = {k: v.to(self.device) for k, v in state["params_star"].items()}
        self.lambda_ewc       = state["lambda_ewc"]
        self._fisher_computed = state["computed"]
        logger.info(f"[EWC] State loaded from {path}")

    @property
    def is_ready(self) -> bool:
        return self._fisher_computed
