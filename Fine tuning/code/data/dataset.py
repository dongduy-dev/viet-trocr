# =============================================================================
# data/dataset.py
# LMDB Dataset, Resize-for-ViT, Pseudo-line Builder, Task-Mixing Collate Fn
# =============================================================================
#
# BUG FIXES in this version:
#
#  [FIX-2a] LMDBDataset no longer reads "datatype-{idx}" keys from LMDB.
#           Those keys do not exist in the exported LMDB files. Instead,
#           the data type is passed as a constructor parameter (default_data_type)
#           and returned verbatim in __getitem__. The caller (main.py) is
#           responsible for instantiating separate datasets with the correct type.
#
#  [FIX-3]  Task-Mixing Isolation in CurriculumCollateFn.__call__:
#           • printed_pool → ALWAYS isolated single words (never concatenated).
#             VinText scene-text backgrounds produce hard seam artifacts when
#             naively concatenated, destroying ViT patch representations.
#           • handwritten_pool → Curriculum concatenation logic (word_ratio).
#             Pseudo-lines are built ONLY from handwritten words using
#             build_pseudo_line() with ±3px jitter and 8-24px random spacing.
#
#  [FIX-4]  smart_resize replaced by resize_for_vit: scale-to-fit within
#           target_h × target_w (128×1536). Never crops — all text preserved.
# =============================================================================

import io
import os
import random
import logging
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as T

logger = logging.getLogger(__name__)


# =============================================================================
# 1.  Resize for ViT — Scale-to-Fit (Never Crops)
# =============================================================================

def resize_for_vit(
    image: Image.Image,
    target_h: int = 128,
    target_w: int = 1536,
    bg_color: Optional[int] = None,
) -> Image.Image:
    """
    Scale-to-fit resize for rectangular ViT input. NEVER crops.

    Replaces the old smart_resize which hard-cropped images with AR > 4.0,
    destroying text data and causing hallucinations.

    Pipeline:
      Step 1 — Compute uniform scale factor so the image fits within
               target_h × target_w. Both dimensions are respected.
      Step 2 — Resize with Bicubic interpolation.
      Step 3 — Pad to exact target_h × target_w with median pixel fill.
               Uses TOP-LEFT alignment: content goes to (0,0), padding
               fills the right and bottom — matching ViT's left-to-right,
               top-to-bottom patch reading order.

    Args:
        image:    Input PIL image (any mode; converted internally).
        target_h: Final output height (e.g. 128).
        target_w: Final output width  (e.g. 1536).
        bg_color: Override background fill value. If None, uses median pixel.

    Returns:
        PIL.Image (RGB, target_w × target_h).
    """
    # Work in grayscale to estimate background without channel complexity
    img_gray = np.array(image.convert("L"), dtype=np.uint8)

    # Use median pixel as the background fill — robust against bright/dark noise
    fill = int(bg_color) if bg_color is not None else int(np.median(img_gray))

    w, h = image.size   # PIL convention: (width, height)

    # ── Step 1: Compute scale to fit within target dimensions ──────────────
    scale_h = target_h / max(h, 1)
    scale_w = target_w / max(w, 1)
    scale = min(scale_h, scale_w)  # Uniform scale — preserve aspect ratio

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # ── Step 2: Resize ─────────────────────────────────────────────────────
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # ── Step 3: Pad to exact target dimensions (top-left aligned) ──────────
    if image.mode == "RGB":
        canvas = Image.new("RGB", (target_w, target_h), (fill, fill, fill))
    else:
        canvas = Image.new("L", (target_w, target_h), fill)

    canvas.paste(image, (0, 0))  # Top-left alignment

    return canvas.convert("RGB")   # TrOCR processor expects RGB


# =============================================================================
# 2.  Pseudo-line Builder — Handwritten Words Only
# =============================================================================

def build_pseudo_line(
    word_images: List[np.ndarray],
    target_h: int = 64,
    spacing_min: int = 8,
    spacing_max: int = 24,
    vertical_jitter: int = 3,
) -> np.ndarray:
    """
    Horizontally concatenate grayscale word images into a pseudo-line.

    This function is ONLY called for handwritten word samples. It bridges
    the gap between Stage 1 (word-level) and Stage 2 (line-level) by giving
    the Decoder exposure to longer sequences without requiring real line data.

    Design choices:
      • Height normalization: All words are resized to target_h (preserving
        their aspect ratio) so the canvas has a uniform height.
      • Vertical jitter ±vertical_jitter px: Simulates the natural baseline
        variation present in real handwritten lines (writers don't write on
        a perfectly flat baseline).
      • Random spacing [spacing_min, spacing_max] px: Simulates natural
        inter-word spacing variation. Uniform spacing would create an
        artificial regularity not present in real data.
      • Median background fill: The canvas is initialized with the median
        pixel value across all words. This creates a seamless background
        and avoids hard edge artifacts at word boundaries.

    Args:
        word_images:     List of grayscale (H × W) uint8 numpy arrays.
        target_h:        Uniform height for all words on the canvas (px).
        spacing_min:     Minimum gap between words (px).
        spacing_max:     Maximum gap between words (px).
        vertical_jitter: Maximum ±px vertical baseline deviation per word.

    Returns:
        Grayscale pseudo-line as a 2D uint8 numpy array of shape (target_h, W).
    """
    if not word_images:
        return np.full((target_h, 10), 255, dtype=np.uint8)

    # ── Step 1: Normalize all words to target_h ────────────────────────────
    resized: List[np.ndarray] = []
    for img in word_images:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            continue
        new_w = max(1, int(w * target_h / h))
        resized.append(
            cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        )

    if not resized:
        return np.full((target_h, 10), 255, dtype=np.uint8)

    # ── Step 2: Compute seamless background color ──────────────────────────
    all_pixels = np.concatenate([img.flatten() for img in resized])
    bg = int(np.median(all_pixels))

    # ── Step 3: Compute canvas dimensions ──────────────────────────────────
    spacings  = [random.randint(spacing_min, spacing_max) for _ in range(len(resized) - 1)]
    total_w   = sum(img.shape[1] for img in resized) + sum(spacings)

    # Extra vertical buffer to absorb jitter without clipping
    canvas_h  = target_h + 2 * vertical_jitter
    canvas    = np.full((canvas_h, total_w), bg, dtype=np.uint8)

    # ── Step 4: Paste words with vertical jitter ──────────────────────────
    x = 0
    for i, img in enumerate(resized):
        jitter  = random.randint(-vertical_jitter, vertical_jitter)
        # Centre word vertically, then apply jitter
        y_start = vertical_jitter + jitter
        y_start = max(0, min(y_start, canvas_h - target_h))  # hard clamp

        word_h, word_w = img.shape[:2]
        canvas[y_start: y_start + word_h, x: x + word_w] = img
        x += word_w
        if i < len(resized) - 1:
            x += spacings[i]

    # ── Step 5: Crop back to target_h (remove jitter buffer) ──────────────
    result = canvas[vertical_jitter: vertical_jitter + target_h, :]
    return result.astype(np.uint8)


# =============================================================================
# 3.  LMDB Dataset
# =============================================================================

class LMDBDataset(Dataset):
    """
    Reads image-label pairs from an LMDB database.

    Expected LMDB key schema (created by 04_export_lmdb.py):
        image-{idx:09d}  → raw JPEG/PNG bytes
        label-{idx:09d}  → UTF-8 text label
        num-samples      → total sample count as ASCII integer

    NOTE — [FIX-2a]:
        There is NO "datatype-{idx}" key in the LMDB files. The previous
        version attempted to read this key and silently returned "unknown"
        when it was missing. This caused the CurriculumCollateFn to treat
        all samples as an unknown domain, disabling Task-Mixing Isolation.

        The fix: data type is supplied by the CALLER via the `default_data_type`
        constructor argument and returned verbatim from __getitem__. The caller
        (main.py) instantiates separate LMDBDataset objects for printed and
        handwritten data and passes the correct type string to each.

    Args:
        lmdb_path:        Path to the LMDB directory.
        target_h:         ViT input height passed to resize_for_vit.
        target_w:         ViT input width passed to resize_for_vit.
        default_data_type: The domain label for ALL samples in this dataset.
                           Pass "printed" or "handwritten" explicitly.
                           The dataset has no way to determine this from the
                           LMDB file itself — the caller must know the source.
        max_samples:      If set, randomly subsample to this many items.
                          Used to build the printed replay buffer in Stage 2b.
        transform:        Optional additional torchvision transform applied
                          AFTER resize_for_vit and BEFORE returning the PIL image.
    """

    _env_cache = {}

    def __init__(
        self,
        lmdb_path: str,
        target_h: int = 128,
        target_w: int = 1536,
        default_data_type: str = "unknown",
        max_samples: Optional[int] = None,
        transform=None,
        keep_raw: bool = False,
    ):
        self.lmdb_path         = lmdb_path
        self.target_h          = target_h
        self.target_w          = target_w
        self.default_data_type = default_data_type 
        self.transform         = transform
        self.keep_raw          = keep_raw

        if lmdb_path not in LMDBDataset._env_cache:
            LMDBDataset._env_cache[lmdb_path] = lmdb.open(
                lmdb_path,
                max_readers=32,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        self.env = LMDBDataset._env_cache[lmdb_path]

        with self.env.begin(write=False) as txn:
            raw = txn.get(b"num-samples")
            if raw is None:
                raise ValueError(f"Missing 'num-samples' in {lmdb_path}")
            total = int(raw.decode("utf-8"))

            # --- AUTO-DETECT KEY FORMAT AND INDEXING ---
            self.key_fmt = "image-{:09d}"
            self.lbl_fmt = "label-{:09d}"
            start_idx = 1

            if txn.get(b"image-000000001") is not None:
                self.key_fmt, start_idx = "image-{:09d}", 1
            elif txn.get(b"image-00000001") is not None:
                self.key_fmt, self.lbl_fmt, start_idx = "image-{:08d}", "label-{:08d}", 1
            elif txn.get(b"image-000000000") is not None:
                self.key_fmt, start_idx = "image-{:09d}", 0
            elif txn.get(b"image-00000000") is not None:
                self.key_fmt, self.lbl_fmt, start_idx = "image-{:08d}", "label-{:08d}", 0

        # Build valid indices based on auto-detected start index
        self.indices = list(range(start_idx, total + start_idx))

        if max_samples is not None and max_samples < len(self.indices):
            self.indices = random.sample(self.indices, max_samples)

        logger.info(
            f"LMDBDataset | path={lmdb_path} | "
            f"data_type={default_data_type} | "
            f"samples={len(self.indices)} | format={self.key_fmt}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Dict[str, Any]:
        idx = self.indices[pos]

        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(self.key_fmt.format(idx).encode())
            lbl_bytes = txn.get(self.lbl_fmt.format(idx).encode())

        if img_bytes is None:
            raise KeyError(f"Missing '{self.key_fmt.format(idx)}' in LMDB '{self.lmdb_path}'")

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # NFC normalization on label — idempotent, ensures LMDB labels
        # and vocab tokens use the same Unicode form before tokenization.
        label = unicodedata.normalize("NFC", lbl_bytes.decode("utf-8")) if lbl_bytes else ""

        # Store raw image before resize for pseudo-line construction (Stage 1).
        # Pseudo-lines need original word dimensions, not 128×1536 padded versions.
        raw_image = image.copy() if self.keep_raw else None

        image = resize_for_vit(image, self.target_h, self.target_w)

        if self.transform is not None:
            image = self.transform(image)

        result = {
            "image":     image,
            "label":     label,
            "data_type": self.default_data_type,
            "idx":       idx,
        }
        if raw_image is not None:
            result["raw_image"] = raw_image
        return result


# =============================================================================
# 4.  Curriculum Collate Function — Task-Mixing Isolation
# =============================================================================

class CurriculumCollateFn:
    """
    Stage 1 collate function with strict Task-Mixing Isolation.

    Core invariant (enforced unconditionally in every phase):
        PRINTED WORDS ARE NEVER CONCATENATED.

    Reasoning:
        VinText scene-text images have complex real-world backgrounds (shop
        signs, street scenes, varying illumination). Concatenating two VinText
        words creates a hard seam at the join boundary where the background
        textures, brightness levels, and color profiles are discontinuous.
        The resulting artifact spans 1–2 ViT patches and appears as a strong
        vertical edge that has no correspondence in real text. If the model
        sees thousands of these artifacts during training it learns to use the
        seam as a "word boundary signal" — a completely spurious feature that
        degrades performance on real line-level data.

        Handwritten words (UIT-HWDB) are rendered against near-uniform
        parchment/paper backgrounds. build_pseudo_line() fills the canvas with
        the median background color, producing smooth transitions that are
        visually indistinguishable from real handwritten lines.

    Pool separation logic:
        The incoming batch already contains a mixture of printed and
        handwritten samples because the DataLoader is built from a ConcatDataset
        of both LMDBDataset objects (see main.py build_stage1_loaders).
        This collate_fn re-separates them by reading sample["data_type"].

    Curriculum (applies ONLY to handwritten_pool):
        Phase 1A — word_ratio=1.0:
            All HW words remain isolated (same as printed). The model learns
            raw visual features. Decoder cross-attention is frozen externally.

        Phase 1B — word_ratio=0.5:
            50% of HW words are grouped into 3-5 word pseudo-lines.
            Decoder starts adapting positional embeddings 8-25.
            Cross-attention unfrozen by Trainer at start of this phase.

        Phase 1C — word_ratio=0.2:
            80% of HW words form 5-7 word pseudo-lines (~15-25 token seqs).
            Near-line-length sequences fully calibrate the Decoder before
            Stage 2 line data is introduced.

    Args:
        processor:   TrOCRProcessor (handles image encoding + label tokenization).
        cfg:         Full config dict from config.yaml.
    """

    def __init__(self, processor, cfg: dict, transform=None):
        self.processor = processor
        self.cfg       = cfg
        self.transform = transform   # Applied to pseudo-lines after construction

        # Initialize to Phase 1A defaults (most conservative)
        ph = cfg["stage1"]["curriculum"]["phase_1a"]
        self.word_ratio  = float(ph["word_ratio"])
        self.concat_min  = int(ph["concat_min"])
        self.concat_max  = int(ph["concat_max"])

        pl = cfg["stage1"]["pseudo_line"]
        self.spacing_min  = int(pl["spacing_min"])
        self.spacing_max  = int(pl["spacing_max"])
        self.jitter       = int(pl["vertical_jitter"])

        self.target_h    = int(cfg["data"]["pseudo_line_height"])
        self.vit_h       = int(cfg["model"]["image_height"])
        self.vit_w       = int(cfg["model"]["image_width"])
        self.max_len     = int(cfg["model"]["max_target_length"])

    def set_phase(self, phase_name: str) -> None:
        """
        Advance the curriculum to a new phase.
        Called by TrOCRTrainer at the start of each epoch.
        """
        ph = self.cfg["stage1"]["curriculum"][phase_name]
        self.word_ratio = float(ph["word_ratio"])
        self.concat_min = int(ph["concat_min"])
        self.concat_max = int(ph["concat_max"])
        logger.info(
            f"[Curriculum] Phase advanced → {phase_name} | "
            f"word_ratio={self.word_ratio:.2f} | "
            f"concat range=[{self.concat_min}, {self.concat_max}]"
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _pil_to_gray_np(pil_img: Image.Image) -> np.ndarray:
        """Convert PIL image (any mode) to grayscale uint8 numpy array."""
        return np.array(pil_img.convert("L"), dtype=np.uint8)

    @staticmethod
    def _gray_np_to_rgb_pil(arr: np.ndarray) -> Image.Image:
        """Convert grayscale numpy array to RGB PIL image."""
        return Image.fromarray(arr, mode="L").convert("RGB")

    def _finalize_batch(
        self,
        images_pil: List[Image.Image],
        labels_str: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Pass a list of PIL images and label strings through the TrOCRProcessor
        to produce the pixel_values and labels tensors expected by the model.
        """
        encoding = self.processor(
            images=images_pil,
            text=labels_str,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        # Replace pad token id with -100 so CrossEntropyLoss ignores pad positions
        labels = encoding["labels"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels
        return encoding

    # ── Main collate logic ─────────────────────────────────────────────────

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process one batch with Task-Mixing Isolation.

        Args:
            batch: List of dicts from LMDBDataset.__getitem__.
                   Each dict has keys: image (PIL), label (str), data_type (str).

        Returns:
            Dict with pixel_values, labels (and optionally attention_mask)
            as PyTorch tensors, ready to be passed directly to the TrOCR model.

        Processing flow:
            1. Split batch into printed_pool and handwritten_pool by data_type.
            2. Process printed_pool: resize_for_vit each image individually.
               Append to final list as isolated words — NO CONCATENATION EVER.
            3. Process handwritten_pool with curriculum:
               a. With probability word_ratio → keep as isolated word.
               b. Otherwise → group with nearby HW samples and call
                  build_pseudo_line(). The pseudo-line image is then
                  resize_for_vit()'d to the ViT input size.
            4. Combine all processed images + labels and pass through processor.
        """
        # ── Stage 1: Pool Separation ───────────────────────────────────────
        # [FIX-3] Critical: separate by data_type before any processing.
        # This is possible only because LMDBDataset now correctly returns
        # the default_data_type set by the caller [FIX-2a].

        printed_pool:     List[Dict] = []
        handwritten_pool: List[Dict] = []

        for sample in batch:
            if sample["data_type"] == "printed":
                printed_pool.append(sample)
            else:
                # "handwritten" or "unknown" both go to HW pool
                # (unknown defaults to handwritten-safe behavior)
                handwritten_pool.append(sample)

        final_images: List[Image.Image] = []
        final_labels: List[str]         = []

        # ── Stage 2: Process Printed Pool — ISOLATION ONLY ────────────────
        # Task-Mixing Isolation invariant: printed words are NEVER concatenated.
        # Rationale: See class docstring. Scene-text backgrounds produce
        # destructive seam artifacts when joined.
        for sample in printed_pool:
            img = sample["image"]
            # resize_for_vit was already applied in __getitem__, but the image
            # may have come from a ConcatDataset without resize (if transform
            # was None). We apply it idempotently here for safety.
            if not isinstance(img, Image.Image):
                img = T.ToPILImage()(img)
            final_images.append(img)
            final_labels.append(sample["label"])

        # ── Stage 3: Process Handwritten Pool — Curriculum Concatenation ───
        # Shuffle HW pool to randomize which samples get concatenated together
        random.shuffle(handwritten_pool)

        used = [False] * len(handwritten_pool)
        i = 0

        while i < len(handwritten_pool):
            if used[i]:
                i += 1
                continue

            sample = handwritten_pool[i]
            used[i] = True

            # ── Decision: isolated word or pseudo-line? ────────────────────
            # word_ratio = probability that this HW sample remains isolated.
            # Phase 1A: word_ratio=1.0 → always isolated
            # Phase 1B: word_ratio=0.5 → 50% chance of isolation
            # Phase 1C: word_ratio=0.2 → 80% chance of concatenation
            if random.random() < self.word_ratio:
                # Keep as isolated word — directly add to final batch
                img = sample["image"]
                if not isinstance(img, Image.Image):
                    img = T.ToPILImage()(img)
                final_images.append(img)
                final_labels.append(sample["label"])

            else:
                # Build a pseudo-line: collect concat_min..concat_max HW words
                n_needed  = random.randint(self.concat_min, self.concat_max)
                # How many additional words can we still grab from the pool?
                available = [
                    j for j in range(i + 1, len(handwritten_pool))
                    if not used[j]
                ]
                # Cap at what is actually available (may be smaller near end of batch)
                n_needed  = min(n_needed, len(available) + 1)  # +1 for current sample

                # Gather the group indices (current + next n_needed-1 available)
                group_indices = [i] + available[: n_needed - 1]
                for gj in group_indices:
                    used[gj] = True

                # Use raw (pre-resize) images for pseudo-line construction.
                # raw_image preserves original word dimensions; avoids the
                # double-resize bug where 128×1536 padded images were
                # downscaled to 64px and re-concatenated.
                word_np_imgs: List[np.ndarray] = []
                for gj in group_indices:
                    src = handwritten_pool[gj].get(
                        "raw_image", handwritten_pool[gj]["image"]
                    )
                    if not isinstance(src, Image.Image):
                        src = T.ToPILImage()(src)
                    word_np_imgs.append(self._pil_to_gray_np(src))

                word_labels: List[str] = [
                    handwritten_pool[gj]["label"] for gj in group_indices
                ]

                # Build pseudo-line canvas with jitter and random spacing
                pseudo_np = build_pseudo_line(
                    word_np_imgs,
                    target_h=self.target_h,
                    spacing_min=self.spacing_min,
                    spacing_max=self.spacing_max,
                    vertical_jitter=self.jitter,
                )

                # Convert canvas back to PIL RGB and apply resize_for_vit
                pseudo_pil = self._gray_np_to_rgb_pil(pseudo_np)
                pseudo_pil = resize_for_vit(
                    pseudo_pil,
                    target_h=self.vit_h,
                    target_w=self.vit_w,
                )

                # Apply training transforms to the assembled pseudo-line
                # (not per-word — rotation and blur are more realistic
                # when applied to the whole line)
                if self.transform is not None:
                    pseudo_pil = self.transform(pseudo_pil)

                final_images.append(pseudo_pil)
                # Label = words joined by space (matches how real lines are labeled)
                final_labels.append(" ".join(word_labels))

            i += 1

        # ── Stage 4: Fallback guard ────────────────────────────────────────
        # If the batch was empty or all samples were filtered out (should not
        # happen in practice but protects against DataLoader edge cases)
        if not final_images:
            logger.warning(
                "[CurriculumCollateFn] Empty batch after processing — "
                "returning raw batch items as fallback."
            )
            for s in batch:
                img = s["image"]
                if not isinstance(img, Image.Image):
                    img = T.ToPILImage()(img)
                final_images.append(img)
                final_labels.append(s["label"])

        # ── Stage 5: Encode through TrOCRProcessor ────────────────────────
        return self._finalize_batch(final_images, final_labels)


# =============================================================================
# 5.  Standard Collate Function — Stage 2 and Stage 3
# =============================================================================

def make_standard_collate(processor, max_len: int = 128):
    """
    Simple collate_fn for stages where no curriculum mixing is needed.
    Images are already PIL (from LMDBDataset) and labels are strings.
    """
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [
            s["image"] if isinstance(s["image"], Image.Image)
            else T.ToPILImage()(s["image"])
            for s in batch
        ]
        texts = [s["label"] for s in batch]

        encoding = processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        labels = encoding["labels"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        encoding["labels"] = labels
        return encoding

    return collate_fn


# =============================================================================
# 6.  Mixed Batch Sampler — Stage 2b (70% HW + 30% Printed Replay)
# =============================================================================

class MixedBatchSampler(torch.utils.data.Sampler):
    """
    Produces batch indices that maintain an EXACT hw_ratio / printed_ratio split
    in every batch, as opposed to WeightedRandomSampler which only guarantees
    the ratio in expectation over many batches.

    Exact-per-batch control matters here because:
      • We want every gradient update in Stage 2b to see ~70% HW gradients.
        With WeightedRandomSampler, some batches could be 95% HW — causing
        a large EWC penalty spike — while others are 50% HW — too little
        handwritten adaptation.
      • The EWC penalty is computed per-batch, so consistent composition
        leads to consistent regularization magnitude.

    Args:
        hw_indices:  Indices belonging to handwritten samples in the ConcatDataset.
        pr_indices:  Indices belonging to printed (replay) samples.
        batch_size:  Total items per batch.
        hw_ratio:    Fraction of each batch that must be handwritten.
        num_batches: Batches per epoch. Defaults to len(hw_indices) // n_hw.
    """

    def __init__(
        self,
        hw_indices:  List[int],
        pr_indices:  List[int],
        batch_size:  int,
        hw_ratio:    float = 0.70,
        num_batches: Optional[int] = None,
    ):
        self.hw_indices  = hw_indices
        self.pr_indices  = pr_indices
        self.n_hw        = max(1, int(batch_size * hw_ratio))
        self.n_pr        = batch_size - self.n_hw
        self.num_batches = num_batches or (len(hw_indices) // self.n_hw)

    def __iter__(self):
        # Shuffle both pools once per epoch
        hw_pool = self.hw_indices.copy()
        pr_pool = self.pr_indices.copy()
        random.shuffle(hw_pool)
        random.shuffle(pr_pool)

        hw_ptr = pr_ptr = 0

        for _ in range(self.num_batches):
            # Wrap with re-shuffle when a pool is exhausted
            if hw_ptr + self.n_hw > len(hw_pool):
                random.shuffle(hw_pool)
                hw_ptr = 0
            if pr_ptr + self.n_pr > len(pr_pool):
                random.shuffle(pr_pool)
                pr_ptr = 0

            batch = (
                hw_pool[hw_ptr: hw_ptr + self.n_hw]
                + pr_pool[pr_ptr: pr_ptr + self.n_pr]
            )
            random.shuffle(batch)   # Intra-batch shuffle for BatchNorm stability
            yield batch

            hw_ptr += self.n_hw
            pr_ptr += self.n_pr

    def __len__(self) -> int:
        return self.num_batches


def build_mixed_dataset_and_loader(
    line_lmdb_path: str,
    processor,
    cfg: dict,
    transform=None,
) -> Tuple[ConcatDataset, DataLoader]:
    """
    Build Stage 2b's mixed LINE DataLoader.

    The LINE LMDB contains both HW and printed samples mixed together.
    Since the LMDB has no datatype key [FIX-2a], we instantiate two separate
    LMDBDataset objects over the SAME physical LMDB but with different
    default_data_type values and filter logic... however, because the LMDB
    truly has no type key, we must rely on the physical separation of the
    underlying LMDB files built during preprocessing.

    For Stage 2b it is assumed that separate HW and printed LINE LMDBs exist
    (analogous to the word-level separation in Stage 1). If only a combined
    LMDB exists, set both hw_lmdb_path and pr_lmdb_path to the same path and
    adjust default_data_type accordingly after verifying ground truth labels.

    See main.py::build_stage2b_loaders for the actual path resolution.

    Args:
        transform: torchvision transform pipeline for training augmentation.
                   Passed to hw_dataset and pr_dataset (train sets only).
                   pr_full is NOT augmented — it is only used to compute the
                   replay buffer size, not for training directly.
    """
    s2b         = cfg["stage2b"]
    target_h    = cfg["model"]["image_height"]
    target_w    = cfg["model"]["image_width"]
    max_len     = cfg["model"]["max_target_length"]

    # These paths are resolved by the caller; passed as a tuple via cfg
    hw_lmdb = cfg["_runtime"]["hw_line_lmdb"]
    pr_lmdb = cfg["_runtime"]["pr_line_lmdb"]

    # NHIỆM VỤ 2: Truyền transform vào hw_dataset và pr_dataset (train sets).
    # pr_full chỉ dùng để tính replay_size — không cần augmentation.
    hw_dataset = LMDBDataset(
        hw_lmdb,
        target_h=target_h,
        target_w=target_w,
        default_data_type="handwritten",
        transform=transform,  # Augmentation: train set
    )

    pr_full = LMDBDataset(
        pr_lmdb,
        target_h=target_h,
        target_w=target_w,
        default_data_type="printed",
        transform=None,  # pr_full chỉ dùng để đếm size, không train trực tiếp
    )
    # Build printed replay buffer as a random subsample of the full printed set
    replay_size = max(1, int(len(pr_full) * s2b["replay_buffer_fraction"]))
    pr_dataset = LMDBDataset(
        pr_lmdb,
        target_h=target_h,
        target_w=target_w,
        default_data_type="printed",
        max_samples=replay_size,
        transform=transform,  # Augmentation: train set
    )

    logger.info(
        f"[Stage2b] HW line samples: {len(hw_dataset)} | "
        f"Printed replay: {len(pr_dataset)} "
        f"({s2b['replay_buffer_fraction']*100:.0f}% of {len(pr_full)})"
    )

    combined   = ConcatDataset([hw_dataset, pr_dataset])
    hw_indices = list(range(len(hw_dataset)))
    pr_indices = list(range(len(hw_dataset), len(hw_dataset) + len(pr_dataset)))

    sampler  = MixedBatchSampler(
        hw_indices=hw_indices,
        pr_indices=pr_indices,
        batch_size=s2b["batch_size"],
        hw_ratio=s2b["handwritten_ratio"],
    )
    collate  = make_standard_collate(processor, max_len)
    loader   = DataLoader(
        combined,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=s2b["num_workers"],
        pin_memory=True,
    )
    return combined, loader
