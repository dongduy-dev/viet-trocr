#!/usr/bin/env python3
"""
app.py — Vietnamese OCR System: Interactive Web Interface
Gradio-based thesis demo with full pipeline visualization.

Usage (Colab):
    %cd /content/trocr_viet/interface
    !python app.py --model_path /content/drive/MyDrive/OCR/checkpoints/final_model

Usage (local):
    python app.py --model_path /path/to/final_model --device cpu
"""

import argparse
import logging
import os
import sys
import time

import gradio as gr
from PIL import Image

# Add current dir to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import OCRPipeline, PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global pipeline instance (initialized on first run or at startup) ──────
_pipeline: OCRPipeline = None
_pipeline_config: dict = {}


def get_or_create_pipeline(
    model_path: str,
    device: str = "cuda",
    batch_size: int = 8,
    enable_phobert: bool = False,
    enable_address: bool = True,
) -> OCRPipeline:
    """Lazy-initialize or reinitialize the pipeline if config changed."""
    global _pipeline, _pipeline_config

    new_config = {
        "model_path": model_path,
        "device": device,
        "batch_size": batch_size,
        "enable_phobert": enable_phobert,
        "enable_address": enable_address,
    }

    if _pipeline is not None and _pipeline_config == new_config:
        return _pipeline

    logger.info("[App] Initializing pipeline with new config...")
    _pipeline = OCRPipeline(
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        enable_phobert=enable_phobert,
        enable_address=enable_address,
    )
    _pipeline_config = new_config
    return _pipeline


def run_ocr(
    image,
    model_path,
    batch_size,
    enable_phobert,
    enable_address,
    progress=gr.Progress(track_tqdm=True),
):
    """Main OCR function called by Gradio."""
    if image is None:
        gr.Warning("Please upload an image first!")
        return [None] * 10

    if not model_path or not model_path.strip():
        gr.Warning("Please set the model path!")
        return [None] * 10

    try:
        pipeline = get_or_create_pipeline(
            model_path=model_path.strip(),
            batch_size=int(batch_size),
            enable_phobert=enable_phobert,
            enable_address=enable_address,
        )
    except Exception as e:
        gr.Warning(f"Pipeline initialization failed: {e}")
        logger.exception("Pipeline init error")
        return [None] * 10

    # Run pipeline
    pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    pil_image = pil_image.convert("RGB")

    try:
        result: PipelineResult = pipeline.run(pil_image, progress_fn=progress)
    except Exception as e:
        gr.Warning(f"Pipeline error: {e}")
        logger.exception("Pipeline run error")
        return [None] * 10

    # ── Format outputs for Gradio ──────────────────────────────────────

    # 1. Final text
    final_text = result.full_text

    # 2. Timing summary
    timing_parts = []
    step_names = {
        "step1_preprocess": "① Preprocessing",
        "step2_detection": "② Detection",
        "step3_segmentation": "③ Segmentation",
        "step4_recognition": "④ Recognition",
        "step5_postprocess": "⑤ Post-processing",
    }
    for key, label in step_names.items():
        t = result.timing.get(key, 0)
        timing_parts.append(f"{label}: {t:.2f}s")
    total = result.timing.get("total", 0)
    timing_str = " │ ".join(timing_parts) + f" │ **Total: {total:.2f}s**"

    # 3. Preprocessed image (Step 1)
    preproc_img = result.preprocessed_image
    deskew_info = f"Deskew angle: {result.deskew_angle:.2f}°"

    # 4. Detection visualization (Step 2)
    det_img = result.annotated_image
    det_info = f"Detected **{result.num_regions}** text regions"

    # 5. Line crops gallery (Step 3)
    line_gallery = []
    for i, img in enumerate(result.line_images):
        line_gallery.append((img, f"Line {i+1}"))
    seg_info = f"Segmented into **{result.num_lines}** text lines"

    # 6. Per-line recognition table (Step 4)
    recognition_data = []
    for i, text in enumerate(result.line_texts):
        recognition_data.append([i + 1, text])

    # 7. Post-processing comparison (Step 5)
    pp = result.post_processed
    post_raw = pp.get("raw", "")
    post_sanitized = pp.get("sanitized", "")
    post_corrected = pp.get("corrected", "")
    post_phobert = pp.get("phobert_corrected", "")

    return (
        final_text,       # output_text
        timing_str,        # timing_display
        preproc_img,       # preproc_image
        deskew_info,       # deskew_info
        det_img,           # detection_image
        det_info,          # detection_info
        line_gallery,      # line_gallery
        seg_info,          # segmentation_info
        recognition_data,  # recognition_table
        post_corrected,    # post_processed_text
    )


# =============================================================================
# Gradio UI
# =============================================================================

CUSTOM_CSS = """
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 800;
    margin-bottom: 0;
    padding: 0;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.05em;
    margin-top: 4px;
}
.timing-bar {
    background: linear-gradient(90deg, #e0e7ff 0%, #c7d2fe 100%);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.92em;
    border-left: 4px solid #6366f1;
}
.step-header {
    font-weight: 700;
    color: #4f46e5;
    font-size: 1.1em;
}
footer { display: none !important; }
"""

def create_ui(default_model_path: str = ""):
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
        title="Vietnamese OCR System",
    ) as demo:

        # ── Header ────────────────────────────────────────────────────
        gr.HTML("""
            <div style="padding: 16px 0 8px 0;">
                <div class="main-title">🔍 Vietnamese OCR System</div>
                <div class="subtitle">
                    End-to-End Pipeline: Preprocessing → DBNet++ → DBSCAN → TrOCR → Post-processing
                </div>
            </div>
        """)

        # ── Input Row ─────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                input_image = gr.Image(
                    label="📤 Upload Document Image",
                    type="pil",
                    height=360,
                )
            with gr.Column(scale=2):
                model_path = gr.Textbox(
                    label="🧠 TrOCR Model Path",
                    value=default_model_path,
                    placeholder="/content/drive/MyDrive/OCR/checkpoints/final_model",
                )
                batch_size = gr.Slider(
                    label="Batch Size",
                    minimum=1, maximum=16, step=1, value=8,
                )
                with gr.Row():
                    enable_phobert = gr.Checkbox(
                        label="Enable PhoBERT",
                        value=False,
                        info="MLM correction (experimental)",
                    )
                    enable_address = gr.Checkbox(
                        label="Address Correction",
                        value=True,
                        info="Gazetteer fuzzy-match",
                    )

                run_btn = gr.Button(
                    "▶  Run OCR Pipeline",
                    variant="primary",
                    size="lg",
                )

        # ── Final Output ──────────────────────────────────────────────
        output_text = gr.Textbox(
            label="📝 Recognized Text (Final Output)",
            lines=6,
            show_copy_button=True,
            interactive=False,
        )

        timing_display = gr.Markdown(
            label="Pipeline Timing",
            elem_classes=["timing-bar"],
        )

        # ── Step-by-Step Visualization ────────────────────────────────
        with gr.Accordion("① Step 1: Preprocessing (CLAHE + Deskew)", open=False):
            with gr.Row():
                preproc_image = gr.Image(label="Enhanced Image", height=300)
                deskew_info = gr.Markdown()

        with gr.Accordion("② Step 2: Text Detection (DBNet++)", open=True):
            detection_info = gr.Markdown()
            detection_image = gr.Image(label="Detected Text Regions", height=400)

        with gr.Accordion("③ Step 3: Line Segmentation (DBSCAN)", open=False):
            segmentation_info = gr.Markdown()
            line_gallery = gr.Gallery(
                label="Cropped Text Lines",
                columns=1,
                height="auto",
                object_fit="contain",
            )

        with gr.Accordion("④ Step 4: Text Recognition (TrOCR)", open=True):
            recognition_table = gr.Dataframe(
                headers=["Line #", "Recognized Text"],
                label="Per-line Recognition Results",
                wrap=True,
            )

        with gr.Accordion("⑤ Step 5: Post-processing", open=False):
            post_processed_text = gr.Textbox(
                label="Corrected Text (Address + Sanitizer)",
                lines=4,
                interactive=False,
            )

        # ── Wire up ──────────────────────────────────────────────────
        run_btn.click(
            fn=run_ocr,
            inputs=[
                input_image,
                model_path,
                batch_size,
                enable_phobert,
                enable_address,
            ],
            outputs=[
                output_text,
                timing_display,
                preproc_image,
                deskew_info,
                detection_image,
                detection_info,
                line_gallery,
                segmentation_info,
                recognition_table,
                post_processed_text,
            ],
        )

        # ── Footer ───────────────────────────────────────────────────
        gr.HTML("""
            <div style="text-align:center; padding:16px; color:#9ca3af; font-size:0.85em;">
                Vietnamese TrOCR OCR System — Graduation Thesis Project<br>
                Pipeline: CLAHE + Hough Deskew → DBNet++ → DBSCAN → TrOCR → PhoBERT/Address Correction
            </div>
        """)

    return demo


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese OCR Web Interface")
    parser.add_argument(
        "--model_path",
        default="/content/drive/MyDrive/OCR/checkpoints/final_model",
        help="Path to exported TrOCR final_model/",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True,
                        help="Create shareable Gradio link")
    args = parser.parse_args()

    demo = create_ui(default_model_path=args.model_path)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
