#!/usr/bin/env python3
# =============================================================================
# main.py — TrOCR Vietnamese Multi-modal Training Entry Point
# =============================================================================
#
# BUG FIXES integrated in this version:
#
#  [FIX-1] pretrained_name = "microsoft/trocr-base-stage1" (see config.yaml)
#
#  [FIX-2] build_stage1_loaders now instantiates TWO separate LMDBDataset
#          objects — one for printed (word_printed_train) and one for
#          handwritten (word_handwritten_train) — each with the correct
#          default_data_type argument. They are combined via ConcatDataset
#          before being passed to the DataLoader.
#          The LMDBDataset no longer reads any "datatype-" keys from LMDB.
#
#  [FIX-3] CurriculumCollateFn enforces Task-Mixing Isolation:
#          printed samples are NEVER concatenated (passed as isolated words);
#          only handwritten samples go through pseudo-line construction.
#          This is correctly triggered by the data_type field now returned
#          reliably by each LMDBDataset via default_data_type [FIX-2].
#
# Positional Embedding Re-initialization is called once in setup_model()
# BEFORE any training stage begins, as required.
#
# Usage (Colab cell):
#   !python main.py --stage all       # Run all stages sequentially
#   !python main.py --stage 1         # Stage 1 only (or resume)
#   !python main.py --stage 2a        # Stage 2a only
#   !python main.py --stage 2b        # Stage 2b only (needs ewc_state.pt)
#   !python main.py --stage 3         # Stage 3 only
# =============================================================================

# ─── COLAB SETUP INSTRUCTIONS ─────────────────────────────────────────────────
# Add the following to your FIRST Colab cell (before importing this script).
# Copying LMDB to local SSD reduces per-sample read latency from ~50ms (Drive)
# to ~1ms (local), which is critical for 150k+ sample random-access datasets.
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#
#   import os
#   os.makedirs('/content/lmdb', exist_ok=True)
#
#   # Copy all LMDB directories at once
#   !cp -r "/content/drive/MyDrive/OCR/lmdb/." "/content/lmdb/"
#   !echo "Done. Contents:" && ls /content/lmdb/
#
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import os
import random
import sys
import unicodedata
from typing import Optional, Tuple
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ── Local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import (
    CurriculumCollateFn,
    LMDBDataset,
    MixedBatchSampler,
    build_mixed_dataset_and_loader,
    make_standard_collate,
    resize_for_vit,
)
from core.ewc     import EWC
from core.trainer import TrOCRTrainer, load_latest_checkpoint


# =============================================================================
# 1.  Logging
# =============================================================================

def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# =============================================================================
# 2.  Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True: cuDNN selects fastest conv algorithm per input shape.
    # Disable for exact reproducibility at the cost of ~5% throughput.
    torch.backends.cudnn.benchmark = True


# =============================================================================
# 3.  ViT Positional Embedding Interpolation
# =============================================================================

def interpolate_vit_pos_embeddings(
    model: VisionEncoderDecoderModel,
    new_h_patches: int = 8,
    new_w_patches: int = 96,
) -> None:
    """
    Interpolate ViT positional embeddings from (24×24) to (new_h × new_w).

    The pre-trained model has 577 pos embeddings:
      index 0     = CLS token (class token positional embedding)
      index 1–576 = 24×24 patch grid

    For 128×1536 input with 16px patches:
      8×96 = 768 patches → 769 total embeddings (1 CLS + 768).

    Uses bicubic interpolation to reshape the 2D grid, preserving
    the learned spatial relationships from pre-training.

    Args:
        model:         VisionEncoderDecoderModel to modify IN PLACE.
        new_h_patches: Number of patch rows for the new resolution.
        new_w_patches: Number of patch columns for the new resolution.
    """
    logger = logging.getLogger(__name__)

    pos_embed = model.encoder.embeddings.position_embeddings  # (1, 577, 768)

    cls_token = pos_embed[:, :1, :]           # (1, 1, 768)
    patch_embed = pos_embed[:, 1:, :]         # (1, 576, 768)

    dim = patch_embed.shape[-1]  # 768
    old_grid = int(patch_embed.shape[1] ** 0.5)  # 24

    logger.info(
        f"[ViT PosEmb] Interpolating positional embeddings: "
        f"({old_grid}×{old_grid}) → ({new_h_patches}×{new_w_patches}) | "
        f"embed_dim={dim}"
    )

    # Reshape to 2D grid → interpolate → flatten
    patch_embed = patch_embed.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
    patch_embed = torch.nn.functional.interpolate(
        patch_embed.float(),
        size=(new_h_patches, new_w_patches),
        mode='bicubic',
        align_corners=False,
    )
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

    # Concatenate CLS + interpolated patches
    new_pos_embed = torch.cat([cls_token.float(), patch_embed], dim=1)
    model.encoder.embeddings.position_embeddings = torch.nn.Parameter(new_pos_embed)

    # Update encoder config AND patch embeddings to reflect new image size.
    # ViTPatchEmbeddings.forward() checks self.image_size — if not updated,
    # it raises ValueError: "Input image size (128*1536) doesn't match model (384*384)".
    new_image_size = (new_h_patches * 16, new_w_patches * 16)  # (128, 1536)
    model.encoder.config.image_size = new_image_size
    model.encoder.embeddings.patch_embeddings.image_size = new_image_size

    new_total = new_pos_embed.shape[1]
    logger.info(
        f"[ViT PosEmb] Done. New positional embeddings shape: "
        f"(1, {new_total}, {dim}) = 1 CLS + {new_h_patches * new_w_patches} patches"
    )


# =============================================================================
# 4.  Model + Tokenizer Setup
# =============================================================================

def setup_model(
    cfg: dict,
    device: torch.device,
) -> Tuple["TrOCRProcessor", "VisionEncoderDecoderModel"]:
    """
    Load TrOCRProcessor and VisionEncoderDecoderModel, then:

      1. Add Vietnamese custom tokens from vocab file.
      2. Resize decoder token embeddings to the new vocab size.
      3. Set special token IDs on model.config.
      4. Interpolate ViT positional embeddings (24×24 → 8×96).

    [FIX-1] Uses 'microsoft/trocr-base-stage1' (config.yaml pretrained_name).
    The stage1 checkpoint is the IIT-CDIP pre-trained base with no language
    or modality bias — the correct foundation before Vietnamese fine-tuning.
    """
    # Import bổ sung để ép dùng Slow Tokenizer
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, RobertaTokenizer

    model_name = cfg["model"]["pretrained_name"]
    vocab_file  = cfg["paths"]["vocab_file"]

    logger = logging.getLogger(__name__)
    logger.info(f"[Setup] Loading base model: {model_name}")
    
    # =================================================================
    # --- FIX: BỎ FAST TOKENIZER ---
    # Ép khởi tạo đích danh bản Python thuần (Slow Tokenizer)
    slow_tokenizer = RobertaTokenizer.from_pretrained(model_name)
    processor = TrOCRProcessor.from_pretrained(model_name)
    processor.tokenizer = slow_tokenizer  # Ghi đè vào processor
    # =================================================================

    # =================================================================
    # --- FIX: DISABLE PROCESSOR'S INTERNAL RESIZE ---
    # The ViTImageProcessor inside TrOCRProcessor defaults to resizing all
    # images to 384×384. Since we handle resizing ourselves via resize_for_vit
    # (128×1536 scale-to-fit with padding), we must tell the processor to
    # accept our pre-resized images as-is.
    #
    # do_resize=False:  Skip the processor's internal resize entirely.
    # size={...}:       Update the expected dimensions for any downstream code
    #                   that reads processor.image_processor.size.
    # =================================================================
    processor.image_processor.do_resize = False
    processor.image_processor.do_normalize = True   # Explicit: use ImageNet mean/std
    processor.image_processor.size = {
        "height": cfg["model"]["image_height"],
        "width":  cfg["model"]["image_width"],
    }
    logger.info(
        f"[Setup] Image processor: do_resize=False, do_normalize=True, "
        f"expected input={cfg['model']['image_height']}×{cfg['model']['image_width']}"
    )
    # =================================================================
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    logger.info(f"[Setup] Tokenizer Class: {type(processor.tokenizer).__name__}")

    # ── Step 1: Add Vietnamese tokens (NFC-normalized) ────────────────────
    # NHIỆM VỤ 1: UNICODE NORMALIZATION
    # Tiếng Việt có hai dạng Unicode cho cùng một ký tự:
    #   NFC (precomposed): 'ộ' = U+1ED9 (1 codepoint)
    #   NFD (decomposed):  'ộ' = 'o' + combining circumflex + combining dot (3 codepoints)
    # Tokenizer chỉ nhận diện đúng một dạng. Nếu vocab file và LMDB labels
    # dùng dạng khác nhau, tokenizer sẽ fallback sang byte-level encoding
    # → CER cao bất thường ngay từ đầu training.
    # Giải pháp: ép toàn bộ tokens về NFC trước khi add, dùng set() để
    # loại bỏ duplicate có thể phát sinh sau khi normalize.
    n_added = 0
    if os.path.exists(vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            raw_tokens = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        # Normalize to NFC và deduplicate
        new_tokens = list(dict.fromkeys(
            unicodedata.normalize("NFC", t) for t in raw_tokens
        ))
        if len(new_tokens) < len(raw_tokens):
            logger.warning(
                f"[Setup] Vocab dedup sau NFC: {len(raw_tokens)} → {len(new_tokens)} tokens."
            )
        n_added = processor.tokenizer.add_tokens(new_tokens)
        logger.info(
            f"[Setup] Vocab file: {len(new_tokens)} tokens | "
            f"Newly added to tokenizer: {n_added}"
        )
        # Validation log: kiểm tra tokenizer nhận diện đúng ký tự tiếng Việt
        test_str = unicodedata.normalize("NFC", "Đại học Tôn Đức Thắng - TrOCR ặ ợ ướ")
        test_ids  = processor.tokenizer.encode(test_str, add_special_tokens=False)
        test_toks = processor.tokenizer.convert_ids_to_tokens(test_ids)
        logger.info(
            f"[Setup] Tokenizer validation:\n"
            f"  Input : {test_str}\n"
            f"  Tokens: {test_toks}\n"
            f"  IDs   : {test_ids}"
        )
    else:
        logger.warning(
            f"[Setup] Vocab file not found: {vocab_file}. "
            "Proceeding with default tokenizer. Vietnamese diacritics may "
            "be tokenized sub-optimally as byte-fallback sequences."
        )

    # ── Step 2: Resize decoder embeddings ─────────────────────────────────
    if n_added > 0:
        new_vocab_size = len(processor.tokenizer)
        model.decoder.resize_token_embeddings(new_vocab_size)
        logger.info(f"[Setup] Decoder embedding resized → vocab_size={new_vocab_size}")

    # ── Step 3: Configure special token IDs & Generation Config ───────────
    # Khai báo biến id để code gọn gàng
    bos_id = processor.tokenizer.bos_token_id
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id

    # 1. Cập nhật model.config (Dành cho kiến trúc mạng và hàm Loss)
    # decoder_start_token_id = eos_id (2) — matches pretrained TrOCR design.
    # The pretrained model was trained with decoder_start=2 (RoBERTa/BART convention).
    # Using bos_id=0 disrupts position 1's pretrained self-attention pattern,
    # causing it to saturate to bos prediction (prob=1.0).
    model.config.decoder_start_token_id = eos_id
    model.config.pad_token_id           = pad_id
    model.config.eos_token_id           = eos_id

    # 2. Cập nhật model.generation_config (BẮT BUỘC cho model.generate())
    model.generation_config.decoder_start_token_id = eos_id
    model.generation_config.pad_token_id           = pad_id
    model.generation_config.eos_token_id           = eos_id
    model.generation_config.use_cache              = True  # CRITICAL: prevent GC contamination
    
    # Thiết lập các tham số Generation — Greedy decoding for eval stability
    model.generation_config.max_length           = cfg["model"]["max_target_length"]
    model.generation_config.length_penalty       = 1.0
    model.generation_config.num_beams            = 1

    # ── Step 4: Interpolate ViT positional embeddings ─────────────────────
    # Reshape pre-trained 24×24 grid to 8×96 for the new 128×1536 input.
    # Called ONCE here, before any training stage.
    interpolate_vit_pos_embeddings(
        model,
        new_h_patches=cfg["model"]["vit_h_patches"],
        new_w_patches=cfg["model"]["vit_w_patches"],
    )

    model = model.to(device)

    # =================================================================
    # ---BUG_FIX: HUGGING FACE META DEVICE TENSORS ---
    # =================================================================
    for module in model.modules():
        if module.__class__.__name__ == "TrOCRSinusoidalPositionalEmbedding":
            if hasattr(module, "weights") and isinstance(module.weights, torch.Tensor):
                n_pos, dim = module.weights.shape
                pad_idx = getattr(module, "padding_idx", None)
                module.weights = module.get_embedding(n_pos, dim, pad_idx).to(device)
                
            if hasattr(module, "_float_tensor") and isinstance(module._float_tensor, torch.Tensor):
                module._float_tensor = torch.zeros(1, device=device)
    # =================================================================

    # CRITICAL: TrOCR's decoder config defaults to use_cache=False.
    # This must be True for model.generate() to properly use KV-cache.
    # Without this, the decoder recomputes all attention at every step
    # and may produce degenerate (all-zero) outputs.
    model.decoder.config.use_cache = True

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    logger.info(
        f"[Setup] Model ready | "
        f"encoder={enc_params:,} params | decoder={dec_params:,} params"
    )
    return processor, model


# =============================================================================
# 5.  Augmentation Pipeline
# =============================================================================

class _MedianFillRotation:
    """RandomRotation that adapts fill color to each image's border pixels.

    Unlike T.RandomRotation(fill=255), this avoids bright-stripe artifacts
    on dark-background images (e.g. VinText scene-text) by matching the
    fill to the padding color from resize_for_vit().
    """

    def __init__(self, degrees: float = 2.0):
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        arr = np.array(img)
        # Median of top and bottom border rows — matches resize_for_vit padding
        border = np.concatenate([arr[0].flatten(), arr[-1].flatten()])
        fill_val = int(np.median(border))
        return img.rotate(
            angle, resample=Image.BICUBIC,
            fillcolor=(fill_val, fill_val, fill_val),
        )


def get_training_transforms() -> T.Compose:
    """
    Lightweight augmentation pipeline applied to training images only.

    Design constraints:
      - Applied AFTER resize_for_vit (128×1536), so transforms work in
        the final ViT input space — consistent pixel density.
      - Label-preserving: no transform changes text content or makes chars
        unrecognizable. Max rotation = 2° keeps diacritics in place.
      - Kept minimal intentionally: TrOCR's ViT encoder is sensitive to
        distribution shift. Aggressive augmentation can harm convergence
        more than it helps for a relatively small Vietnamese OCR dataset.

    NOT applied to:
      - Validation / test sets (never — would corrupt eval metrics)
      - fisher_loader (EWC Fisher must reflect the true printed distribution)
    """
    return T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
        _MedianFillRotation(degrees=2.0),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))],
            p=0.3,
        ),
    ])


# =============================================================================
# 6.  DataLoader Builders
# =============================================================================

def _lmdb_path(cfg: dict, key: str) -> str:
    """Resolve a full LMDB path from config: local_lmdb_root + lmdb[key]."""
    return os.path.join(cfg["paths"]["local_lmdb_root"], cfg["paths"]["lmdb"][key])


def _build_val_loader(
    cfg: dict,
    lmdb_key: str,
    default_data_type: str,
    processor,
) -> Optional[DataLoader]:
    """
    Build a validation DataLoader for a single domain.

    [FIX-2] Passes default_data_type explicitly — no LMDB key reading.
    Returns None if the LMDB path does not exist or the dataset is empty.
    """
    path = _lmdb_path(cfg, lmdb_key)
    if not os.path.exists(path):
        logger.warning(f"[Val] LMDB not found, skipping: {path}")
        return None

    ds = LMDBDataset(
        path,
        target_h=cfg["model"]["image_height"],
        target_w=cfg["model"]["image_width"],
        default_data_type=default_data_type,
    )
    if len(ds) == 0:
        logger.warning(f"[Val] Dataset empty ({path}), skipping.")
        return None

    collate = make_standard_collate(processor, cfg["model"]["max_target_length"])
    return DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True,
    )


def build_stage1_loaders(
    cfg: dict,
    processor,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], CurriculumCollateFn]:
    """
    Build Stage 1 DataLoaders.

    [FIX-2] Core change: Instead of one combined word LMDB dataset, we
    instantiate TWO separate LMDBDataset objects:

      printed_word_ds:     word_printed_train LMDB
                           default_data_type = "printed"
                           → CurriculumCollateFn will NEVER concatenate these.

      handwritten_word_ds: word_handwritten_train LMDB
                           default_data_type = "handwritten"
                           → CurriculumCollateFn applies curriculum concat logic.

    They are merged via ConcatDataset so the DataLoader treats them as one
    dataset. The data_type field in each sample tells the collate_fn which
    pool a sample belongs to.

    This is the prerequisite for [FIX-3] Task-Mixing Isolation to work correctly.
    """
    s1_cfg   = cfg["stage1"]
    target_h  = cfg["model"]["image_height"]
    target_w  = cfg["model"]["image_width"]
    max_len   = cfg["model"]["max_target_length"]

    # ── [FIX-2] Two separate datasets with explicit data_type labels ───────

    printed_path = _lmdb_path(cfg, "word_printed_train")
    hw_path      = _lmdb_path(cfg, "word_handwritten_train")

    if not os.path.exists(printed_path):
        raise FileNotFoundError(
            f"Printed word LMDB not found: {printed_path}\n"
            "Expected path from config: paths.lmdb.word_printed_train\n"
            "Run 04_export_lmdb.py with word-level separation enabled."
        )
    if not os.path.exists(hw_path):
        raise FileNotFoundError(
            f"Handwritten word LMDB not found: {hw_path}\n"
            "Expected path from config: paths.lmdb.word_handwritten_train\n"
            "Run 04_export_lmdb.py with word-level separation enabled."
        )

    train_transform = get_training_transforms()

    printed_word_ds = LMDBDataset(
        printed_path,
        target_h=target_h,
        target_w=target_w,
        default_data_type="printed",     # Task-Mixing: this pool → isolation only
        transform=train_transform,
    )

    handwritten_word_ds = LMDBDataset(
        hw_path,
        target_h=target_h,
        target_w=target_w,
        default_data_type="handwritten", # Task-Mixing: this pool → curriculum concat
        transform=train_transform,
        keep_raw=True,   # [C3-FIX] Preserve raw image for pseudo-line construction
    )

    logger.info(
        f"[Stage1] Printed word samples:     {len(printed_word_ds):,}"
    )
    logger.info(
        f"[Stage1] Handwritten word samples: {len(handwritten_word_ds):,}"
    )

    # ── Merge into one dataset for DataLoader ─────────────────────────────
    combined_word_ds = ConcatDataset([printed_word_ds, handwritten_word_ds])

    # ── CurriculumCollateFn — Task-Mixing Isolation ───────────────────────
    # transform is passed so pseudo-lines get augmented after construction
    collate_fn = CurriculumCollateFn(processor, cfg, transform=train_transform)
    # Trainer will call collate_fn.set_phase() at each epoch boundary.

    train_loader = DataLoader(
        combined_word_ds,
        batch_size=s1_cfg["batch_size"],
        shuffle=True,      # Shuffle ensures printed and HW samples are interspersed
        num_workers=s1_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,    # Avoids partial batches that could skew pool separation
    )

    # ── Validation loaders — LINE level (line-level val = realistic benchmark) ─
    # We validate on line-level data even during Stage 1 so we can detect early
    # if the pseudo-line curriculum is actually bridging toward real line CER.
    # Each val loader reads from its own physically separated LMDB directory.
    val_printed_loader     = _build_val_loader(cfg, "line_printed_val",     "printed",     processor)
    val_handwritten_loader = _build_val_loader(cfg, "line_handwritten_val", "handwritten", processor)

    return train_loader, val_printed_loader, val_handwritten_loader, collate_fn


def build_stage2a_loaders(
    cfg: dict,
    processor,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], DataLoader]:
    """
    Build Stage 2a DataLoaders — printed LINE data only.

    Returns:
        train_loader, val_printed_loader, val_handwritten_loader, fisher_loader
        fisher_loader is a smaller-batch version of train_loader for EWC computation.
    """
    s2a_cfg  = cfg["stage2a"]
    target_h  = cfg["model"]["image_height"]
    target_w  = cfg["model"]["image_width"]
    max_len   = cfg["model"]["max_target_length"]

    # Stage 2a trains ONLY on printed LINE data.
    # Using the physically separated line_printed_train LMDB ensures zero
    # handwritten samples contaminate this stage, which would blur the
    # Fisher Information Matrix computed at the end of Stage 2a.
    line_path = _lmdb_path(cfg, "line_printed_train")
    if not os.path.exists(line_path):
        raise FileNotFoundError(
            f"Line printed train LMDB not found: {line_path}\n"
            "Expected config key: paths.lmdb.line_printed_train\n"
            "Rebuild LMDB with domain-separated line-level export."
        )

    # [FIX-2] default_data_type="printed" — no LMDB key reading
    pr_train_ds = LMDBDataset(
        line_path,
        target_h=target_h,
        target_w=target_w,
        default_data_type="printed",
        transform=get_training_transforms(),  # Augmentation: train set only
    )

    collate = make_standard_collate(processor, max_len)

    train_loader = DataLoader(
        pr_train_ds,
        batch_size=s2a_cfg["batch_size"],
        shuffle=True,
        num_workers=s2a_cfg["num_workers"],
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )

    # Fisher loader: NO augmentation — must reflect true printed distribution.
    # Augmented gradients would produce a biased Fisher matrix, causing EWC
    # to protect wrong weights. Use a fresh dataset instance with transform=None.
    pr_fisher_ds = LMDBDataset(
        line_path,
        target_h=target_h,
        target_w=target_w,
        default_data_type="printed",
        transform=None,  # NO augmentation for Fisher
    )
    fisher_loader = DataLoader(
        pr_fisher_ds,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True,
    )

    val_printed_loader     = _build_val_loader(cfg, "line_printed_val",     "printed",     processor)
    val_handwritten_loader = _build_val_loader(cfg, "line_handwritten_val", "handwritten", processor)

    return train_loader, val_printed_loader, val_handwritten_loader, fisher_loader


def build_stage2b_loaders(
    cfg: dict,
    processor,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Build Stage 2b DataLoaders — Mixed HW + Printed Replay.

    Now resolves TWO physically separate LMDB paths:
      hw_line_path → line_handwritten_train  (70% of every batch)
      pr_line_path → line_printed_train      (30% replay, 15% subsample)

    Both paths are injected into cfg["_runtime"] so that
    build_mixed_dataset_and_loader() in dataset.py can read them without
    a signature change (preserving dataset.py / trainer.py / ewc.py intact).
    """
    # Resolve both domain-separated LMDB paths independently
    hw_line_path = _lmdb_path(cfg, "line_handwritten_train")
    pr_line_path = _lmdb_path(cfg, "line_printed_train")

    if not os.path.exists(hw_line_path):
        raise FileNotFoundError(
            f"HW Line LMDB not found: {hw_line_path}\n"
            "Expected config key: paths.lmdb.line_handwritten_train"
        )
    if not os.path.exists(pr_line_path):
        raise FileNotFoundError(
            f"PR Line LMDB not found: {pr_line_path}\n"
            "Expected config key: paths.lmdb.line_printed_train"
        )

    # Inject runtime paths into cfg for build_mixed_dataset_and_loader.
    # This avoids modifying that function's signature (which would cascade
    # changes into dataset.py, breaking the "do not modify" constraint).
    cfg.setdefault("_runtime", {})
    cfg["_runtime"]["hw_line_lmdb"] = hw_line_path
    cfg["_runtime"]["pr_line_lmdb"] = pr_line_path

    # Pass None as the first positional arg — the function reads paths from
    # cfg["_runtime"] when the first arg is None.
    _, mixed_loader = build_mixed_dataset_and_loader(
        None, processor, cfg, transform=get_training_transforms()
    )

    val_printed_loader     = _build_val_loader(cfg, "line_printed_val",     "printed",     processor)
    val_handwritten_loader = _build_val_loader(cfg, "line_handwritten_val", "handwritten", processor)

    return mixed_loader, val_printed_loader, val_handwritten_loader


def build_stage3_loaders(
    cfg: dict,
    processor,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build Stage 3 DataLoaders — Paragraph level."""
    s3_cfg   = cfg["stage3"]
    target_h  = cfg["model"]["image_height"]
    target_w  = cfg["model"]["image_width"]
    max_len   = cfg["model"]["max_target_length"]

    para_path = _lmdb_path(cfg, "paragraph_train")
    if not os.path.exists(para_path):
        raise FileNotFoundError(f"Paragraph train LMDB not found: {para_path}")

    # [FIX-2] Paragraph data is handwritten-only in this project (no printed paragraphs)
    para_ds = LMDBDataset(
        para_path,
        target_h=target_h,
        target_w=target_w,
        default_data_type="handwritten",
    )
    collate = make_standard_collate(processor, max_len)

    train_loader = DataLoader(
        para_ds,
        batch_size=s3_cfg["batch_size"],
        shuffle=True,
        num_workers=s3_cfg["num_workers"],
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = _build_val_loader(cfg, "paragraph_val", "handwritten", processor)
    return train_loader, val_loader


# =============================================================================
# 6.  Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrOCR Vietnamese Multi-modal Training"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "1", "2a", "2b", "3"],
        help=(
            "Training stage to run. Use 'all' for full pipeline. "
            "Individual stages will auto-resume from the latest checkpoint."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ── Logging ───────────────────────────────────────────────────────────
    setup_logging(cfg["paths"]["log_dir"])
    logger.info("=" * 70)
    logger.info(
        "TrOCR Vietnamese | stage=%s | model=%s",
        args.stage,
        cfg["model"]["pretrained_name"],
    )
    logger.info("=" * 70)

    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(cfg["training"]["seed"])

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB"
        )
    else:
        logger.warning("No GPU detected — training will be very slow on CPU.")

    # ── Google Drive Mount (Colab only) ───────────────────────────────────
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        logger.info("Google Drive mounted at /content/drive")
    except ImportError:
        logger.info("Not in Colab — skipping Drive mount.")

    # ── Verify local LMDB exists ──────────────────────────────────────────
    local_lmdb = cfg["paths"]["local_lmdb_root"]
    if not os.path.exists(local_lmdb) or not os.listdir(local_lmdb):
        logger.error(
            f"Local LMDB directory is empty or missing: {local_lmdb}\n"
            "Run the following BEFORE executing main.py:\n"
            f"  !mkdir -p {local_lmdb}\n"
            f"  !cp -r \"{cfg['paths']['gdrive_lmdb_root']}/.\" \"{local_lmdb}/\""
        )
        sys.exit(1)

    # ── Model + Processor ─────────────────────────────────────────────────
    # [FIX-1] model name is "microsoft/trocr-base-stage1" from config.
    # [FIX-2] Positional embedding interpolation is called inside setup_model().
    processor, model = setup_model(cfg, device)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = TrOCRTrainer(model, processor, cfg, device)

    run_all = (args.stage == "all")

    # =========================================================================
    # Stage 1 — Curriculum Word → Pseudo-line
    # =========================================================================
    # [FIX-2] build_stage1_loaders creates TWO separate LMDBDataset objects
    #         (printed + handwritten) and merges them with ConcatDataset.
    # [FIX-3] CurriculumCollateFn.Task-Mixing Isolation is enabled by the
    #         correct data_type values returned by each LMDBDataset.
    if run_all or args.stage == "1":
        logger.info("─" * 70)
        logger.info("STAGE 1 — Curriculum Word → Pseudo-line")
        logger.info("─" * 70)

        (
            s1_train_loader,
            s1_val_printed,
            s1_val_hw,
            collate_fn,
        ) = build_stage1_loaders(cfg, processor)

        trainer.train_stage1(
            train_loader=s1_train_loader,
            val_printed_loader=s1_val_printed,
            val_handwritten_loader=s1_val_hw,
            collate_fn=collate_fn,
        )

    # =========================================================================
    # Stage 2a — Printed LINE Fine-tuning + Fisher Information Computation
    # =========================================================================
    ewc: Optional[EWC] = None

    if run_all or args.stage == "2a":
        logger.info("─" * 70)
        logger.info("STAGE 2a — Printed LINE Fine-tuning")
        logger.info("─" * 70)

        # ── Load Stage 1 weights nếu chưa có Stage 2a checkpoint ──────────
        import glob
        s2a_existing = glob.glob(
            os.path.join(cfg["paths"]["checkpoint_dir"], "stage2a_epoch_*.pt")
        )
        if not s2a_existing:
            logger.info("[Stage2a] No stage2a checkpoint found. Loading Stage 1 weights...")
            # Try best checkpoint first, then fall back to latest epoch
            s1_best = os.path.join(cfg["paths"]["checkpoint_dir"], "stage1_best.pt")
            if os.path.exists(s1_best):
                state = torch.load(s1_best, map_location=device)
                model.load_state_dict(state["model_state"])
                logger.info(f"[Stage2a] Loaded Stage 1 BEST checkpoint from {s1_best}")
            else:
                loaded_epoch, _, _ = load_latest_checkpoint(
                    cfg["paths"]["checkpoint_dir"],
                    "stage1",
                    model,
                    device=device,
                )
                if loaded_epoch == 0:
                    logger.warning(
                        "[Stage2a] Stage 1 checkpoint not found! "
                        "Training will start from base pretrained weights. "
                        "Verify stage1_best.pt or stage1_epoch_*.pt exists in checkpoint_dir."
                    )
                else:
                    logger.info(f"[Stage2a] Stage 1 weights loaded (epoch {loaded_epoch}). Proceeding.")
        # ──────────────────────────────────────────────────────────────────

        (
            s2a_train_loader,
            s2a_val_printed,
            s2a_val_hw,
            fisher_loader,
        ) = build_stage2a_loaders(cfg, processor)

        ewc = trainer.train_stage2a(
            train_loader=s2a_train_loader,
            val_printed_loader=s2a_val_printed,
            val_handwritten_loader=s2a_val_hw,
            fisher_loader=fisher_loader,
        )

    # =========================================================================
    # Stage 2b — Mixed HW + Printed Replay with EWC
    # =========================================================================
    if run_all or args.stage == "2b":
        logger.info("─" * 70)
        logger.info("STAGE 2b — Mixed HW + Printed Replay (EWC)")
        logger.info("─" * 70)

        # ── [C1-FIX] Load Stage 2a weights if no Stage 2b checkpoint ──────
        # Without this, running `--stage 2b` after a Colab restart starts
        # from the base pretrained model, wasting all Stage 1 + 2a progress
        # and producing invalid EWC penalties.
        import glob as _glob
        s2b_existing = _glob.glob(
            os.path.join(cfg["paths"]["checkpoint_dir"], "stage2b_epoch_*.pt")
        )
        if not s2b_existing:
            logger.info("[Stage2b] No stage2b checkpoint found. Loading Stage 2a weights...")
            s2a_best = os.path.join(cfg["paths"]["checkpoint_dir"], "stage2a_best.pt")
            if os.path.exists(s2a_best):
                state = torch.load(s2a_best, map_location=device)
                model.load_state_dict(state["model_state"])
                logger.info(f"[Stage2b] Loaded Stage 2a BEST checkpoint from {s2a_best}")
            else:
                loaded_epoch, _, _ = load_latest_checkpoint(
                    cfg["paths"]["checkpoint_dir"],
                    "stage2a",
                    model,
                    device=device,
                )
                if loaded_epoch == 0:
                    logger.warning(
                        "[Stage2b] Stage 2a checkpoint not found! "
                        "Training will start from base pretrained weights. "
                        "EWC will be ineffective."
                    )
                else:
                    logger.info(f"[Stage2b] Stage 2a weights loaded (epoch {loaded_epoch}).")

        # [M3-FIX] EWC loading is handled inside trainer.train_stage2b()
        # — removed the redundant loading block that was here.

        (
            s2b_mixed_loader,
            s2b_val_printed,
            s2b_val_hw,
        ) = build_stage2b_loaders(cfg, processor)

        trainer.train_stage2b(
            mixed_loader=s2b_mixed_loader,
            val_printed_loader=s2b_val_printed,
            val_handwritten_loader=s2b_val_hw,
            ewc=ewc,
        )

    # =========================================================================
    # Stage 3 — Paragraph Adaptation (Encoder Frozen)
    # =========================================================================
    if run_all or args.stage == "3":
        logger.info("─" * 70)
        logger.info("STAGE 3 — Paragraph Adaptation")
        logger.info("─" * 70)

        if not cfg["stage3"]["enabled"]:
            logger.info("[Stage3] Disabled in config, skipping.")
        else:
            s3_train_loader, s3_val_loader = build_stage3_loaders(cfg, processor)
            trainer.train_stage3(
                train_loader=s3_train_loader,
                val_loader=s3_val_loader,
            )

    # =========================================================================
    # Save Final Model
    # =========================================================================
    if run_all:
        final_path = os.path.join(cfg["paths"]["checkpoint_dir"], "final_model")
        os.makedirs(final_path, exist_ok=True)
        model.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        logger.info(f"Final model saved → {final_path}")

    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
