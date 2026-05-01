# =============================================================================
# core/trainer.py
# Custom PyTorch Training Loop — No HuggingFace Trainer API
# Handles: LLRD, AMP, Grad Accumulation, EWC, Dual Validation, Auto-Resume
# =============================================================================

import os
import glob
import math
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrOCRProcessor
import jiwer
from tqdm import tqdm

from core.ewc import EWC

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Layer-wise Learning Rate Decay (LLRD)
# =============================================================================

def build_llrd_optimizer(
    model: nn.Module,
    cfg: dict,
    stage_cfg_key: str,
) -> AdamW:
    """
    Create an AdamW optimizer with Layer-wise Learning Rate Decay.

    Motivation:
      Low-level encoder layers learn primitive visual features (edges, strokes).
      These are already good from pre-training. High LR here would destroy them.
      The decoder generates text and needs to adapt faster.

    Strategy:
      - Encoder layers 0..split_layer-1  → lr * encoder_low_multiplier
      - Encoder layers split_layer..end   → lr * encoder_top_multiplier
      - Decoder all layers                → lr (base rate)
      - Cross-attention specifically      → lr * 0.5 (intermediate)

    Args:
        model:          TrOCR model (VisionEncoderDecoderModel).
        cfg:            Full config dict.
        stage_cfg_key:  e.g. "stage1", "stage2a", "stage2b".
    """
    stage_lr   = cfg[stage_cfg_key]["lr"]
    base_lr    = stage_lr["decoder_base"]
    low_mult   = stage_lr["encoder_low_multiplier"]
    top_mult   = stage_lr["encoder_top_multiplier"]
    split      = stage_lr["encoder_split_layer"]

    param_groups = []

    # ── Encoder ─────────────────────────────────────────────────────────────
    encoder = model.encoder
    encoder_layers = getattr(encoder, "encoder", encoder)  # ViT has .encoder.layer
    all_encoder_layers = getattr(encoder_layers, "layer", [])

    for layer_idx, layer in enumerate(all_encoder_layers):
        mult = low_mult if layer_idx < split else top_mult
        param_groups.append({
            "params": [p for p in layer.parameters() if p.requires_grad],
            "lr":     base_lr * mult,
            "name":   f"encoder_layer_{layer_idx}",
        })

    # Encoder embedding / patch projection (treated as lowest layers)
    encoder_other = [
        p for name, p in encoder.named_parameters()
        if p.requires_grad and not any(
            f"layer.{i}." in name for i in range(len(all_encoder_layers))
        )
    ]
    if encoder_other:
        param_groups.append({
            "params": encoder_other,
            "lr":     base_lr * low_mult,
            "name":   "encoder_embeddings",
        })

    # ── Decoder ──────────────────────────────────────────────────────────────
    decoder = model.decoder
    decoder_layers = getattr(
        getattr(decoder, "model", decoder),
        "decoder",
        decoder,
    )
    decoder_layer_list = getattr(decoder_layers, "layers", [])

    for layer_idx, layer in enumerate(decoder_layer_list):
        # Separate cross-attention from self-attention for finer control
        cross_attn_params = []
        other_params = []
        for name, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder_attn" in name or "cross_attn" in name:
                cross_attn_params.append(p)
            else:
                other_params.append(p)

        if cross_attn_params:
            param_groups.append({
                "params": cross_attn_params,
                "lr":     base_lr * 0.5,   # Cross-attention: intermediate LR
                "name":   f"decoder_cross_attn_{layer_idx}",
            })
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr":     base_lr,
                "name":   f"decoder_layer_{layer_idx}",
            })

    # Decoder embedding + output projection
    decoder_other = []
    for name, p in decoder.named_parameters():
        if not p.requires_grad:
            continue
        in_layer = any(
            f"layers.{i}." in name
            for i in range(len(decoder_layer_list))
        )
        if not in_layer:
            decoder_other.append(p)

    if decoder_other:
        param_groups.append({
            "params": decoder_other,
            "lr":     base_lr,
            "name":   "decoder_embeddings_and_head",
        })

    # Remove empty groups (safeguard)
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = AdamW(param_groups, weight_decay=0.01)

    # Log LR assignment summary
    for g in param_groups:
        logger.debug(
            f"  LLRD group={g['name']} | lr={g['lr']:.2e} | "
            f"params={sum(p.numel() for p in g['params'])}"
        )

    total_params = sum(
        p.numel() for g in param_groups for p in g["params"]
    )
    logger.info(
        f"[LLRD] Optimizer built | groups={len(param_groups)} | "
        f"trainable_params={total_params:,} | base_lr={base_lr:.2e}"
    )
    return optimizer


# =============================================================================
# 2. Warmup + Cosine Decay Scheduler
# =============================================================================

def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay to zero.

    This is the standard schedule for fine-tuning: avoids large gradient
    steps at the beginning (when the model is far from equilibrium) and
    smoothly reduces LR at the end to allow fine convergence.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / max(1, warmup_steps)
        # Cosine decay from warmup_steps to total_steps
        progress = float(current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# 3. Metrics: CER and WER via jiwer
# =============================================================================

def compute_cer_wer(
    predictions: List[str],
    references: List[str],
) -> Tuple[float, float]:
    """
    Compute Character Error Rate and Word Error Rate.

    Returns:
        (CER, WER) as floats in [0, 1].
    """
    if not predictions or not references:
        return 1.0, 1.0

    # jiwer requires non-empty strings; replace empty with a space
    preds = [p if p.strip() else " " for p in predictions]
    refs  = [r if r.strip() else " " for r in references]

    cer = jiwer.cer(refs, preds)
    wer = jiwer.wer(refs, preds)
    return float(cer), float(wer)


# =============================================================================
# 4. Checkpoint Utilities
# =============================================================================

def save_checkpoint(
    path: str,
    stage: str,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    metrics: dict,
    ewc: Optional[EWC] = None,
    ewc_path: Optional[str] = None,
    best_cer: Optional[float] = None,
) -> None:
    """Save full training state for exact resume after Colab restart."""
    state = {
        "stage":        stage,
        "epoch":        epoch,
        "global_step":  global_step,
        "model_state":  model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "scaler":       scaler.state_dict(),
        "metrics":      metrics,
        "best_cer":     best_cer,  # Persists best CER across Colab restarts
    }
    torch.save(state, path)

    # Save EWC state separately (it can be large — Fisher matrices)
    if ewc is not None and ewc_path is not None:
        ewc.save(ewc_path)

    logger.info(f"[Checkpoint] Saved: {path}")


def load_latest_checkpoint(
    checkpoint_dir: str,
    stage: str,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = None,
) -> Tuple[int, int, dict]:
    """
    Auto-resume: find the latest .pt file for this stage and load it.

    Returns:
        (start_epoch, global_step, metrics) — (0, 0, {}) if no checkpoint found.
    """
    pattern = os.path.join(checkpoint_dir, f"{stage}_epoch_*.pt")
    files   = sorted(glob.glob(pattern))

    if not files:
        logger.info(f"[Checkpoint] No checkpoint found for stage={stage}. Starting fresh.")
        return 0, 0, {}

    latest = files[-1]
    state  = torch.load(latest, map_location=device or "cpu")

    model.load_state_dict(state["model_state"])
    if optimizer  is not None: optimizer.load_state_dict(state["optimizer"])
    if scheduler  is not None: scheduler.load_state_dict(state["scheduler"])
    if scaler     is not None: scaler.load_state_dict(state["scaler"])

    logger.info(
        f"[Checkpoint] Resumed from {latest} | "
        f"epoch={state['epoch']} | step={state['global_step']}"
    )
    return state["epoch"] + 1, state["global_step"], state.get("metrics", {})


def cleanup_old_checkpoints(checkpoint_dir: str, stage: str, keep_n: int = 3) -> None:
    """Remove older checkpoints, keeping only the latest keep_n files."""
    pattern = os.path.join(checkpoint_dir, f"{stage}_epoch_*.pt")
    files   = sorted(glob.glob(pattern))
    if len(files) > keep_n:
        for old in files[:-keep_n]:
            os.remove(old)
            logger.debug(f"[Checkpoint] Removed old checkpoint: {old}")


# =============================================================================
# 5. Evaluation Loop (Dual: Printed + Handwritten)
# =============================================================================

def safe_batch_decode(tokenizer, token_ids_tensor):
    """
    Bypass lỗi cắt byte (byte truncation) của HuggingFace đối với AddedTokens.
    """
    special_ids = {tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, -100}
    results = []
    for seq in token_ids_tensor:
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        
        # Lọc bỏ padding và special tokens
        clean_ids = [idx for idx in seq if idx not in special_ids]
        
        # Map trực tiếp từ ID sang chuỗi character gốc một cách an toàn
        tokens = tokenizer.convert_ids_to_tokens(clean_ids)
        
        # RoBERTa dùng 'Ġ' để đại diện cho dấu cách. Chuyển nó về khoảng trắng bình thường.
        text = "".join(tokens).replace("Ġ", " ").replace("Ċ", "\n")
        
        results.append(text.strip())
    return results

@torch.no_grad()
def evaluate(
    model: nn.Module,
    processor: TrOCRProcessor,
    printed_loader: Optional[DataLoader],
    handwritten_loader: Optional[DataLoader],
    device: torch.device,
    max_new_tokens: int = 128,
    fp16: bool = True,
) -> Dict[str, float]:
    """
    Run evaluation on separate printed and handwritten validation sets.

    Returns dict with keys:
        printed_cer, printed_wer, handwritten_cer, handwritten_wer
    """
    model.eval()
    # CRITICAL: Ensure use_cache=True for generation. gradient_checkpointing
    # sets decoder.config.use_cache=False and GC disable doesn't restore it
    # for VisionEncoderDecoderModel's nested decoder config.
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'config'):
        model.decoder.config.use_cache = True
    results = {}

    loaders = {
        "printed":     printed_loader,
        "handwritten": handwritten_loader,
    }

    for split_name, loader in loaders.items():
        if loader is None:
            logger.warning(f"[Eval] No loader for {split_name}, skipping.")
            continue

        all_preds = []
        all_refs  = []
        first_batch_logged = False

        for batch in tqdm(loader, desc=f"Eval {split_name}", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=fp16):
                generated = model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )

            # Debug: log raw generated token IDs for the first batch
            if not first_batch_logged:
                sample_ids = generated[0].tolist()
                logger.info(
                    f"[Eval DEBUG] {split_name} first sample raw IDs "
                    f"(len={len(sample_ids)}): {sample_ids[:30]}..."
                )
                first_batch_logged = True

            preds = safe_batch_decode(processor.tokenizer, generated)

            # Decode references (replace -100 with pad_token_id before decode)
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = (
                processor.tokenizer.pad_token_id
            )
            refs = safe_batch_decode(processor.tokenizer, labels_for_decode)

            all_preds.extend(preds)
            all_refs.extend(refs)

        logger.info(f"\n[DEBUG EVAL - {split_name.upper()}]")
        
        total_samples = len(all_refs)
        if total_samples > 0:
            # Chọn 5 điểm neo (Anchor points) rải rác: Đầu, 1/4, Giữa, 3/4, và Cuối
            anchor_indices = [
                0, 
                total_samples // 4, 
                total_samples // 2, 
                (3 * total_samples) // 4, 
                total_samples - 1
            ]
            # Loại bỏ index trùng lặp (nếu tập val quá nhỏ) và sắp xếp lại
            anchor_indices = sorted(list(set([idx for idx in anchor_indices if idx < total_samples])))
            
            for i, idx in enumerate(anchor_indices):
                logger.info(f"Mẫu {i+1} [idx={idx:03d}] - Ref : {all_refs[idx]}")
                logger.info(f"Mẫu {i+1} [idx={idx:03d}] - Pred: {all_preds[idx]}")
        logger.info("-" * 60 + "\n")
        # =========================================================

        # =========================================================
        # LOG: TOP 5 MẪU TỆ NHẤT (HARD NEGATIVES)
        # =========================================================
        logger.info(f"\n[HARD NEGATIVES - {split_name.upper()} - TOP 5 TỆ NHẤT]")
        
        # Bước 1: Tính CER riêng lẻ cho từng mẫu để xếp hạng
        individual_metrics = []
        for idx, (ref, pred) in enumerate(zip(all_refs, all_preds)):
            # Đảm bảo chuỗi không rỗng để tránh lỗi thư viện jiwer
            safe_ref = ref if ref.strip() else " "
            safe_pred = pred if pred.strip() else " "
            try:
                single_cer = jiwer.cer(safe_ref, safe_pred)
            except Exception:
                single_cer = 1.0 # Fallback an toàn
            individual_metrics.append((single_cer, idx, ref, pred))
            
        # Bước 2: Sắp xếp danh sách giảm dần theo mức độ tệ của CER
        individual_metrics.sort(key=lambda x: x[0], reverse=True)
        
        # Bước 3: Trích xuất và in ra top 5 mẫu có lỗi lớn nhất
        top_k = min(5, len(individual_metrics))
        for i in range(top_k):
            c, idx, ref, pred = individual_metrics[i]
            logger.info(f"Top {i+1} [idx={idx:03d} | CER={c:.4f}]")
            logger.info(f"  Ref : {ref}")
            logger.info(f"  Pred: {pred}")
        logger.info("-" * 60 + "\n")
        # =========================================================
        
        # Tính toán tổng thể CER/WER cho toàn tập Validation
        cer, wer = compute_cer_wer(all_preds, all_refs)
        results[f"{split_name}_cer"] = cer
        results[f"{split_name}_wer"] = wer
        logger.info(
            f"[Eval] {split_name.upper()} | CER={cer:.4f} | WER={wer:.4f}"
        )

    model.train()
    return results


# =============================================================================
# 6. Core Training Step
# =============================================================================

def training_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    scaler: torch.amp.GradScaler,
    optimizer: AdamW,
    scheduler: LambdaLR,
    ewc: Optional[EWC],
    accumulation_steps: int,
    step_in_accum: int,            # 0-indexed within accumulation window
    max_grad_norm: float,
    fp16: bool,
) -> Tuple[float, float]:
    """
    Perform a single forward + backward step with gradient accumulation.

    Returns:
        (ce_loss_item, ewc_loss_item) for logging.
        Optimizer.step() is called only when step_in_accum == accumulation_steps-1.
    """
    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
    labels       = batch["labels"].to(device, non_blocking=True)

    with torch.amp.autocast("cuda", enabled=fp16):
        outputs  = model(pixel_values=pixel_values, labels=labels)
        ce_loss  = outputs.loss / accumulation_steps  # Normalize for accumulation

        ewc_loss_val = torch.tensor(0.0, device=device)
        if ewc is not None and ewc.is_ready:
            # EWC penalty IS divided by accumulation_steps: each micro-batch
            # adds EWC/N to the gradient, and N micro-batches accumulate to
            # give EWC total — matching the intended lambda magnitude.
            ewc_loss_val = ewc.ewc_loss(model) / accumulation_steps
            total_loss = ce_loss + ewc_loss_val
        else:
            total_loss = ce_loss

    scaler.scale(total_loss).backward()

    # Only update weights after accumulation_steps backward passes
    is_update_step = (step_in_accum == accumulation_steps - 1)
    if is_update_step:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # Chỉ advance scheduler nếu optimizer thực sự step
        # (scaler giữ nguyên scale = không có inf/nan = optimizer đã step)
        if scaler.get_scale() >= scale_before:
            scheduler.step()
        optimizer.zero_grad()

    return ce_loss.item() * accumulation_steps, ewc_loss_val.item() * accumulation_steps


# =============================================================================
# 7. Stage Trainers
# =============================================================================

class TrOCRTrainer:
    """
    Orchestrates all training stages for Vietnamese TrOCR.

    Responsibilities:
      - Stage 1: Curriculum word/pseudo-line training
      - Stage 2a: Printed line fine-tuning + Fisher computation
      - Stage 2b: Mixed handwritten+printed replay with EWC
      - Stage 3: Paragraph adaptation (encoder frozen)
      - Auto-resume from Drive checkpoints
      - Dual validation (printed CER + handwritten CER)
      - Early stopping based on printed CER regression
    """

    def __init__(
        self,
        model: nn.Module,
        processor: TrOCRProcessor,
        cfg: dict,
        device: torch.device,
    ):
        self.model     = model
        self.processor = processor
        self.cfg       = cfg
        self.device    = device
        self.fp16      = cfg["training"]["fp16"]
        self.scaler    = torch.amp.GradScaler("cuda", enabled=self.fp16)

        self.ckpt_dir  = cfg["paths"]["checkpoint_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1
    # ─────────────────────────────────────────────────────────────────────────

    def train_stage1(
        self,
        train_loader: DataLoader,
        val_printed_loader: Optional[DataLoader],
        val_handwritten_loader: Optional[DataLoader],
        collate_fn,                   # CurriculumCollateFn instance
    ) -> None:
        """
        Stage 1: Curriculum Word → Pseudo-line training.

        Curriculum phases (from config):
          Phase 1A: Pure words only → Encoder learns character-level features
          Phase 1B: 50% words, 50% pseudo-lines (3-5 words) → Bridge
          Phase 1C: 20% words, 80% pseudo-lines (5-7 words) → Line readiness

        Decoder cross-attention is frozen during 1A to prevent it from
        learning word-level language statistics that don't generalize to lines.
        """
        stage_cfg = self.cfg["stage1"]
        total_epochs = stage_cfg["epochs"]
        accum_steps  = stage_cfg["accumulation_steps"]

        optimizer = build_llrd_optimizer(self.model, self.cfg, "stage1")
        total_steps = (len(train_loader) // accum_steps) * total_epochs
        scheduler = build_scheduler(optimizer, stage_cfg["warmup_steps"], total_steps)

        # Resume from checkpoint if available
        start_epoch, global_step, prev_metrics = load_latest_checkpoint(
            self.ckpt_dir, "stage1", self.model, optimizer, scheduler,
            self.scaler, self.device
        )

        # Determine if decoder cross-attention should start frozen
        freeze_until = stage_cfg.get("freeze_decoder_until_phase", "phase_1b")
        self._set_decoder_cross_attn_frozen(
            frozen=(freeze_until == "phase_1b" and start_epoch < 5)
        )

        # Restore best CER from latest checkpoint (persists across Colab restarts)
        best_printed_cer = float("inf")
        if prev_metrics:  # prev_metrics comes from load_latest_checkpoint
            # Check top-level best_cer in the checkpoint dict
            latest_ckpts = sorted(glob.glob(
                os.path.join(self.ckpt_dir, "stage1_epoch_*.pt")
            ))
            if latest_ckpts:
                latest_state = torch.load(latest_ckpts[-1], map_location=self.device)
                best_printed_cer = latest_state.get("best_cer", float("inf")) or float("inf")
        # Fallback: read from stage1_best.pt
        if best_printed_cer == float("inf"):
            s1_best_path = os.path.join(self.ckpt_dir, "stage1_best.pt")
            if os.path.exists(s1_best_path):
                s1_best_state = torch.load(s1_best_path, map_location=self.device)
                best_printed_cer = s1_best_state.get("best_cer", float("inf")) or float("inf")
                if best_printed_cer < float("inf"):
                    logger.info(f"[Stage1] Restored best CER={best_printed_cer:.4f} from stage1_best.pt")

        log_every = self.cfg["training"]["log_every_steps"]
        # Use per-stage eval frequency override if available
        eval_every = stage_cfg.get("eval_every_epochs",
                                   self.cfg["training"]["eval_every_epochs"])
        save_every = self.cfg["training"]["save_every_epochs"]

        logger.info(f"[Stage1] Starting from epoch {start_epoch}/{total_epochs} | best_cer={best_printed_cer:.4f}")

        for epoch in range(start_epoch, total_epochs):
            # ── Advance curriculum phase ──────────────────────────────────
            phase = self._get_curriculum_phase(epoch, stage_cfg["curriculum"])
            collate_fn.set_phase(phase)

            # Unfreeze decoder cross-attention at Phase 1B
            if phase == "phase_1b":
                self._set_decoder_cross_attn_frozen(frozen=False)

            self.model.train()

            # Enable gradient checkpointing to fit batch=32 with 768 patches
            # (128×1536 input). Without GC, cross-attention activations across
            # 12 decoder layers exceed L4's 22GB VRAM.
            self.model.gradient_checkpointing_enable()

            epoch_ce_loss = 0.0
            n_batches = 0
            t0 = time.time()

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Stage1 E{epoch}")):
                step_in_accum = batch_idx % accum_steps

                ce_loss, _ = training_step(
                    model=self.model,
                    batch=batch,
                    device=self.device,
                    scaler=self.scaler,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ewc=None,
                    accumulation_steps=accum_steps,
                    step_in_accum=step_in_accum,
                    max_grad_norm=self.cfg["training"]["max_grad_norm"],
                    fp16=self.fp16,
                )

                epoch_ce_loss += ce_loss
                n_batches += 1
                global_step += 1

                if global_step % log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        f"[Stage1] E{epoch} step={global_step} | "
                        f"CE={ce_loss:.4f} | LR={lr_now:.2e} | "
                        f"phase={phase}"
                    )

            avg_loss = epoch_ce_loss / max(1, n_batches)
            elapsed  = time.time() - t0

            # Disable GC before eval (eval uses torch.no_grad, no checkpointing needed)
            self.model.gradient_checkpointing_disable()
            # CRITICAL: GC sets decoder.config.use_cache=False. GC disable
            # does NOT restore it for nested VisionEncoderDecoderModel decoder.
            # Without this, model.generate() cannot use KV-cache → all zeros.
            self.model.decoder.config.use_cache = True

            logger.info(
                f"[Stage1] Epoch {epoch} done | "
                f"avg_loss={avg_loss:.4f} | time={elapsed:.0f}s"
            )

            # ── Evaluation ────────────────────────────────────────────────
            metrics = {}
            if epoch % eval_every == 0:
                # NOTE: Previously skipped eval in phase_1a. Removed so we can
                # verify position 1 is learning with unfrozen cross-attention.
                metrics = evaluate(
                    self.model, self.processor,
                    val_printed_loader, val_handwritten_loader,
                    self.device, fp16=self.fp16
                )
                # Track best CER and save stage1_best.pt
                curr_cer = metrics.get("printed_cer", float("inf"))
                if curr_cer < best_printed_cer:
                    best_printed_cer = curr_cer
                    best_path = os.path.join(self.ckpt_dir, "stage1_best.pt")
                    save_checkpoint(
                        best_path, "stage1", epoch, global_step,
                        self.model, optimizer, scheduler, self.scaler,
                        metrics, best_cer=best_printed_cer,
                    )
                    logger.info(f"[Stage1] New best printed CER={best_printed_cer:.4f}")

            # ── Save checkpoint ───────────────────────────────────────────
            if epoch % save_every == 0:
                ckpt_path = os.path.join(
                    self.ckpt_dir, f"stage1_epoch_{epoch:03d}.pt"
                )
                save_checkpoint(
                    ckpt_path, "stage1", epoch, global_step,
                    self.model, optimizer, scheduler, self.scaler, metrics,
                    best_cer=best_printed_cer,
                )
                cleanup_old_checkpoints(
                    self.ckpt_dir, "stage1",
                    self.cfg["training"]["keep_last_n_checkpoints"]
                )

        logger.info("[Stage1] Complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2a
    # ─────────────────────────────────────────────────────────────────────────

    def train_stage2a(
        self,
        train_loader: DataLoader,
        val_printed_loader: Optional[DataLoader],
        val_handwritten_loader: Optional[DataLoader],
        fisher_loader: Optional[DataLoader] = None,
    ) -> Optional[EWC]:
        """
        Stage 2a: Printed LINE fine-tuning.

        Encoder bottom layers are frozen at a very low LR to preserve
        Stage 1 visual features. Decoder adapts to Vietnamese printed text
        and builds the language prior needed for Stage 2b.

        After training, optionally computes Fisher Information Matrix for EWC.
        """
        stage_cfg   = self.cfg["stage2a"]
        total_epochs = stage_cfg["epochs"]
        accum_steps  = stage_cfg["accumulation_steps"]

        optimizer = build_llrd_optimizer(self.model, self.cfg, "stage2a")
        total_steps = (len(train_loader) // accum_steps) * total_epochs
        scheduler = build_scheduler(optimizer, stage_cfg["warmup_steps"], total_steps)

        start_epoch, global_step, _ = load_latest_checkpoint(
            self.ckpt_dir, "stage2a", self.model, optimizer, scheduler,
            self.scaler, self.device
        )

        # Restore best CER from latest checkpoint (persists across Colab restarts)
        # First try the latest epoch checkpoint's best_cer field
        best_printed_cer = float("inf")
        latest_ckpts = sorted(glob.glob(
            os.path.join(self.ckpt_dir, "stage2a_epoch_*.pt")
        ))
        if latest_ckpts:
            latest_state = torch.load(latest_ckpts[-1], map_location=self.device)
            best_printed_cer = latest_state.get("best_cer", float("inf")) or float("inf")
            if best_printed_cer < float("inf"):
                logger.info(f"[Stage2a] Restored best CER={best_printed_cer:.4f} from epoch checkpoint")
        # Fallback: read from stage2a_best.pt
        if best_printed_cer == float("inf"):
            s2a_best_path = os.path.join(self.ckpt_dir, "stage2a_best.pt")
            if os.path.exists(s2a_best_path):
                best_state = torch.load(s2a_best_path, map_location=self.device)
                best_printed_cer = best_state.get("best_cer", float("inf")) or float("inf")
                if best_printed_cer == float("inf"):
                    # Legacy fallback: read from metrics
                    best_printed_cer = best_state.get("metrics", {}).get("printed_cer", float("inf"))
                if best_printed_cer < float("inf"):
                    logger.info(f"[Stage2a] Restored best CER={best_printed_cer:.4f} from stage2a_best.pt")

        log_every  = self.cfg["training"]["log_every_steps"]
        eval_every = self.cfg["training"]["eval_every_epochs"]
        save_every = self.cfg["training"]["save_every_epochs"]

        logger.info(f"[Stage2a] Starting from epoch {start_epoch}/{total_epochs} | best_cer={best_printed_cer:.4f}")

        for epoch in range(start_epoch, total_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Stage2a E{epoch}")):
                step_in_accum = batch_idx % accum_steps
                ce_loss, _ = training_step(
                    self.model, batch, self.device, self.scaler,
                    optimizer, scheduler, ewc=None,
                    accumulation_steps=accum_steps,
                    step_in_accum=step_in_accum,
                    max_grad_norm=self.cfg["training"]["max_grad_norm"],
                    fp16=self.fp16,
                )
                epoch_loss += ce_loss
                n_batches  += 1
                global_step += 1

                if global_step % log_every == 0:
                    logger.info(
                        f"[Stage2a] E{epoch} step={global_step} | "
                        f"CE={ce_loss:.4f} | LR={scheduler.get_last_lr()[0]:.2e}"
                    )

                # if global_step >= 20:
                #     break

            logger.info(
                f"[Stage2a] Epoch {epoch} | avg_loss={epoch_loss/max(1,n_batches):.4f}"
            )

            metrics = {}
            if epoch % eval_every == 0:
                metrics = evaluate(
                    self.model, self.processor,
                    val_printed_loader, val_handwritten_loader,
                    self.device, fp16=self.fp16
                )
                curr_cer = metrics.get("printed_cer", float("inf"))
                if curr_cer < best_printed_cer:
                    best_printed_cer = curr_cer
                    best_path = os.path.join(self.ckpt_dir, "stage2a_best.pt")
                    save_checkpoint(
                        best_path, "stage2a", epoch, global_step,
                        self.model, optimizer, scheduler, self.scaler,
                        metrics, best_cer=best_printed_cer,
                    )
                    logger.info(f"[Stage2a] New best printed CER={best_printed_cer:.4f}")

            if epoch % save_every == 0:
                ckpt_path = os.path.join(self.ckpt_dir, f"stage2a_epoch_{epoch:03d}.pt")
                save_checkpoint(
                    ckpt_path, "stage2a", epoch, global_step,
                    self.model, optimizer, scheduler, self.scaler, metrics,
                    best_cer=best_printed_cer,
                )
                cleanup_old_checkpoints(self.ckpt_dir, "stage2a",
                    self.cfg["training"]["keep_last_n_checkpoints"])

        logger.info("[Stage2a] Complete.")

        # ── Compute Fisher Information Matrix ─────────────────────────────
        ewc = None
        if stage_cfg.get("compute_fisher", False):
            logger.info("[Stage2a] Computing Fisher Information for EWC...")

            # ==========================================
            # FIX: XÓA OPTIMIZER ĐỂ GIẢI PHÓNG 3GB VRAM
            # ==========================================
            try:
                del optimizer
                del scheduler
                # Xóa biến batch cũ nếu nó còn sót lại từ vòng lặp
                del batch 
            except NameError:
                pass

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # ==========================================

            ewc = EWC(self.model, self.cfg["stage2b"]["ewc"]["lambda"], self.device)
            
            # ==========================================
            # BATCH SIZE 1
            # ==========================================
            if fisher_loader is not None:
                ewc_loader = fisher_loader
            else:
                from torch.utils.data import DataLoader
                ewc_loader = DataLoader(
                    dataset=train_loader.dataset,
                    batch_size=1,  
                    shuffle=True,
                    num_workers=train_loader.num_workers,
                    collate_fn=train_loader.collate_fn
                )
            
            # =======================================================
            # BẬT GRADIENT CHECKPOINTING
            # =======================================================
            logger.info("[Stage2a] Turn on Gradient Checkpointing...")
            self.model.gradient_checkpointing_enable()

            ewc.compute_fisher(
                ewc_loader,
                num_samples=stage_cfg["fisher_samples"],
                fp16=self.fp16,
            )
            ewc_path = os.path.join(self.ckpt_dir, "ewc_state.pt")
            ewc.save(ewc_path)

            # TẮT GRADIENT CHECKPOINTING
            self.model.gradient_checkpointing_disable()
            self.model.decoder.config.use_cache = True

        return ewc

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2b
    # ─────────────────────────────────────────────────────────────────────────

    def train_stage2b(
        self,
        mixed_loader: DataLoader,
        val_printed_loader: Optional[DataLoader],
        val_handwritten_loader: Optional[DataLoader],
        ewc: Optional[EWC] = None,
    ) -> None:
        """
        Stage 2b: Mixed Handwritten + Printed Replay with EWC.

        Key mechanisms:
          1. MixedBatchSampler enforces 70% HW / 30% printed per batch.
          2. EWC penalty prevents overwriting Stage 2a printed knowledge.
          3. LLRD keeps encoder almost frozen; decoder adapts to handwriting.
          4. Early stopping if printed_cer regresses by > 2% absolute.
          5. Sliding window smoothing on HW CER to avoid false early stops
             from noisy small val set (549 samples).
        """
        stage_cfg   = self.cfg["stage2b"]
        total_epochs = stage_cfg["epochs"]
        accum_steps  = stage_cfg["accumulation_steps"]
        delta_limit  = stage_cfg["early_stop_printed_cer_delta"]
        window_size  = stage_cfg["cer_sliding_window"]

        # Try to load EWC state from disk if not passed (Colab resume)
        if ewc is None and stage_cfg["ewc"]["enabled"]:
            ewc_path = os.path.join(self.ckpt_dir, "ewc_state.pt")
            if os.path.exists(ewc_path):
                ewc = EWC(
                    self.model,
                    stage_cfg["ewc"]["lambda"],
                    self.device
                )
                ewc.load(ewc_path)
                logger.info("[Stage2b] EWC state loaded from disk.")
            else:
                logger.warning(
                    "[Stage2b] EWC enabled but no ewc_state.pt found. "
                    "Training without EWC — run Stage 2a first."
                )

        optimizer = build_llrd_optimizer(self.model, self.cfg, "stage2b")
        # NOTE: len(mixed_loader) = MixedBatchSampler.__len__() = batches/epoch.
        # Each batch is already complete, so // accum_steps = optimizer steps/epoch.
        total_steps = (len(mixed_loader) // accum_steps) * total_epochs
        scheduler = build_scheduler(optimizer, stage_cfg["warmup_steps"], total_steps)

        start_epoch, global_step, prev_metrics = load_latest_checkpoint(
            self.ckpt_dir, "stage2b", self.model, optimizer, scheduler,
            self.scaler, self.device
        )

        # Baseline printed CER from Stage 2a best — used for early stopping guard
        best_2a_cer = float("inf")
        # Try stage2a_best.pt first (most reliable — this is the actual best)
        best_2a_path = os.path.join(self.ckpt_dir, "stage2a_best.pt")
        if os.path.exists(best_2a_path):
            state = torch.load(best_2a_path, map_location=self.device)
            best_2a_cer = state.get("best_cer", float("inf")) or float("inf")
            if best_2a_cer == float("inf"):
                # Legacy fallback: read from metrics
                best_2a_cer = state.get("metrics", {}).get("printed_cer", float("inf"))
        logger.info(f"[Stage2b] Baseline printed CER from 2a = {best_2a_cer:.4f}")

        # Sliding window buffer for handwritten CER (smooths noisy small val)
        hw_cer_window: deque = deque(maxlen=window_size)

        # ── Best HW CER tracking ──────────────────────────────────────────
        best_hw_cer = float("inf")
        # Try to restore from latest epoch checkpoint
        latest_ckpts = sorted(glob.glob(
            os.path.join(self.ckpt_dir, "stage2b_epoch_*.pt")
        ))
        if latest_ckpts:
            latest_state = torch.load(latest_ckpts[-1], map_location=self.device)
            best_hw_cer = latest_state.get("best_cer", float("inf")) or float("inf")
            if best_hw_cer < float("inf"):
                logger.info(f"[Stage2b] Restored best HW CER={best_hw_cer:.4f} from epoch checkpoint")
        # Fallback: read from stage2b_best.pt
        if best_hw_cer == float("inf"):
            s2b_best_path = os.path.join(self.ckpt_dir, "stage2b_best.pt")
            if os.path.exists(s2b_best_path):
                best_state = torch.load(s2b_best_path, map_location=self.device)
                best_hw_cer = best_state.get("best_cer", float("inf")) or float("inf")
                if best_hw_cer == float("inf"):
                    best_hw_cer = best_state.get("metrics", {}).get("handwritten_cer", float("inf"))
                if best_hw_cer < float("inf"):
                    logger.info(f"[Stage2b] Restored best HW CER={best_hw_cer:.4f} from stage2b_best.pt")

        log_every  = self.cfg["training"]["log_every_steps"]
        eval_every = self.cfg["training"]["eval_every_epochs"]
        save_every = self.cfg["training"]["save_every_epochs"]

        logger.info(f"[Stage2b] Starting from epoch {start_epoch}/{total_epochs} | best_hw_cer={best_hw_cer:.4f}")

        for epoch in range(start_epoch, total_epochs):
            self.model.train()

            # =======================================================
            # FIX: BẬT GRADIENT CHECKPOINTING CHO STAGE 2B
            # =======================================================
            logger.info("[Stage2b] Turn on Gradient Checkpointing...")
            self.model.gradient_checkpointing_enable()
            # =======================================================

            epoch_ce  = 0.0
            epoch_ewc = 0.0
            n_batches  = 0

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(mixed_loader, desc=f"Stage2b E{epoch}")):
                step_in_accum = batch_idx % accum_steps
                ce_loss, ewc_loss = training_step(
                    self.model, batch, self.device, self.scaler,
                    optimizer, scheduler, ewc=ewc,
                    accumulation_steps=accum_steps,
                    step_in_accum=step_in_accum,
                    max_grad_norm=self.cfg["training"]["max_grad_norm"],
                    fp16=self.fp16,
                )
                epoch_ce  += ce_loss
                epoch_ewc += ewc_loss
                n_batches  += 1
                global_step += 1

                if global_step % log_every == 0:
                    logger.info(
                        f"[Stage2b] E{epoch} step={global_step} | "
                        f"CE={ce_loss:.4f} | EWC={ewc_loss:.4f} | "
                        f"LR={scheduler.get_last_lr()[0]:.2e}"
                    )

                # if global_step >= 20:
                #     break

            # =======================================================
            # TẮT SAU KHI HOÀN THÀNH EPOCH (Để an toàn cho Eval)
            # =======================================================
            self.model.gradient_checkpointing_disable()
            self.model.decoder.config.use_cache = True
            # =======================================================
            
            logger.info(
                f"[Stage2b] Epoch {epoch} | "
                f"avg_CE={epoch_ce/max(1,n_batches):.4f} | "
                f"avg_EWC={epoch_ewc/max(1,n_batches):.4f}"
            )

            metrics = {}
            if epoch % eval_every == 0:
                metrics = evaluate(
                    self.model, self.processor,
                    val_printed_loader, val_handwritten_loader,
                    self.device, fp16=self.fp16
                )

                curr_pr_cer = metrics.get("printed_cer", float("inf"))
                curr_hw_cer = metrics.get("handwritten_cer", float("inf"))

                # ── Sliding window for HW CER ─────────────────────────────
                hw_cer_window.append(curr_hw_cer)
                hw_cer_smooth = sum(hw_cer_window) / len(hw_cer_window)
                logger.info(
                    f"[Stage2b] Smoothed HW CER (window={window_size}): "
                    f"{hw_cer_smooth:.4f}"
                )

                # ── Save best checkpoint if HW CER improved ───────────────
                # Only save if printed CER hasn't regressed beyond safety
                printed_safe = (
                    best_2a_cer == float("inf")
                    or curr_pr_cer <= best_2a_cer + delta_limit
                )
                if curr_hw_cer < best_hw_cer and printed_safe:
                    best_hw_cer = curr_hw_cer
                    best_path = os.path.join(self.ckpt_dir, "stage2b_best.pt")
                    save_checkpoint(
                        best_path, "stage2b", epoch, global_step,
                        self.model, optimizer, scheduler, self.scaler,
                        metrics, best_cer=best_hw_cer,
                    )
                    logger.info(
                        f"[Stage2b] New best HW CER={best_hw_cer:.4f} "
                        f"(printed={curr_pr_cer:.4f})"
                    )

                # ── Early stopping on printed CER regression ──────────────
                if best_2a_cer < float("inf"):
                    regression = curr_pr_cer - best_2a_cer
                    if regression > delta_limit:
                        logger.warning(
                            f"[Stage2b] EARLY STOP: printed CER regressed "
                            f"by {regression:.4f} > threshold {delta_limit}. "
                            f"Current={curr_pr_cer:.4f}, Baseline={best_2a_cer:.4f}"
                        )
                        # Save final checkpoint before stopping
                        ckpt_path = os.path.join(
                            self.ckpt_dir, f"stage2b_epoch_{epoch:03d}_early_stop.pt"
                        )
                        save_checkpoint(
                            ckpt_path, "stage2b", epoch, global_step,
                            self.model, optimizer, scheduler, self.scaler, metrics
                        )
                        return

            if epoch % save_every == 0:
                ckpt_path = os.path.join(self.ckpt_dir, f"stage2b_epoch_{epoch:03d}.pt")
                save_checkpoint(
                    ckpt_path, "stage2b", epoch, global_step,
                    self.model, optimizer, scheduler, self.scaler, metrics,
                    best_cer=best_hw_cer,
                    ewc=ewc,
                    ewc_path=os.path.join(self.ckpt_dir, "ewc_state.pt"),
                )
                cleanup_old_checkpoints(self.ckpt_dir, "stage2b",
                    self.cfg["training"]["keep_last_n_checkpoints"])

        logger.info("[Stage2b] Complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3
    # ─────────────────────────────────────────────────────────────────────────

    def train_stage3(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
    ) -> None:
        """
        Stage 3: Paragraph-level adaptation.

        Encoder is fully frozen — only Decoder cross-attention adapts to
        handle long multi-line sequences. Very low LR to avoid regression.
        """
        if not self.cfg["stage3"]["enabled"]:
            logger.info("[Stage3] Disabled in config, skipping.")
            return

        stage_cfg   = self.cfg["stage3"]
        total_epochs = stage_cfg["epochs"]
        accum_steps  = stage_cfg["accumulation_steps"]

        # Freeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        logger.info("[Stage3] Encoder frozen.")

        optimizer = build_llrd_optimizer(self.model, self.cfg, "stage3")
        total_steps = (len(train_loader) // accum_steps) * total_epochs
        scheduler = build_scheduler(optimizer, stage_cfg["warmup_steps"], total_steps)

        start_epoch, global_step, _ = load_latest_checkpoint(
            self.ckpt_dir, "stage3", self.model, optimizer, scheduler,
            self.scaler, self.device
        )

        log_every  = self.cfg["training"]["log_every_steps"]
        save_every = self.cfg["training"]["save_every_epochs"]

        for epoch in range(start_epoch, total_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0

            optimizer.zero_grad()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Stage3 E{epoch}")):
                step_in_accum = batch_idx % accum_steps
                ce_loss, _ = training_step(
                    self.model, batch, self.device, self.scaler,
                    optimizer, scheduler, ewc=None,
                    accumulation_steps=accum_steps,
                    step_in_accum=step_in_accum,
                    max_grad_norm=self.cfg["training"]["max_grad_norm"],
                    fp16=self.fp16,
                )
                epoch_loss += ce_loss
                n_batches  += 1
                global_step += 1

                if global_step % log_every == 0:
                    logger.info(
                        f"[Stage3] E{epoch} step={global_step} | "
                        f"CE={ce_loss:.4f}"
                    )

                # if global_step >= 20:
                #     break
                
            logger.info(
                f"[Stage3] Epoch {epoch} | avg_loss={epoch_loss/max(1,n_batches):.4f}"
            )

            if epoch % save_every == 0:
                ckpt_path = os.path.join(self.ckpt_dir, f"stage3_epoch_{epoch:03d}.pt")
                save_checkpoint(
                    ckpt_path, "stage3", epoch, global_step,
                    self.model, optimizer, scheduler, self.scaler, {}
                )
                cleanup_old_checkpoints(self.ckpt_dir, "stage3",
                    self.cfg["training"]["keep_last_n_checkpoints"])

        # Unfreeze encoder after stage3 (restore for inference or further tuning)
        for param in self.model.encoder.parameters():
            param.requires_grad = True

        logger.info("[Stage3] Complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_curriculum_phase(self, epoch: int, curriculum: dict) -> str:
        """Determine which curriculum phase applies to the current epoch."""
        for phase_name, phase_cfg in curriculum.items():
            if phase_cfg["start"] <= epoch < phase_cfg["end"]:
                return phase_name
        # Default to last phase
        return list(curriculum.keys())[-1]

    def _set_decoder_cross_attn_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze cross-attention layers in the decoder."""
        for name, param in self.model.decoder.named_parameters():
            if "encoder_attn" in name or "cross_attn" in name:
                param.requires_grad = not frozen
        state = "FROZEN" if frozen else "UNFROZEN"
        logger.info(f"[Trainer] Decoder cross-attention: {state}")
