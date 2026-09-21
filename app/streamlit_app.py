"""
Streamlit web application for Cityscapes semantic segmentation.

Features:
- Upload any street image and get a semantic segmentation prediction
- Side-by-side comparison: original image vs. segmentation overlay
- Adjustable overlay opacity slider
- Color-coded class legend with all 19 Cityscapes classes
- Per-class IoU display (loaded from evaluation results if available)
- Loading spinner during inference

Usage:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Fix import path — allow importing from src/
# ---------------------------------------------------------------------------
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC_DIR)

from cityscapes_dataset import (
    CLASS_NAMES,
    CITYSCAPES_COLORMAP,
    colorize_mask,
    overlay_mask,
)
from config import IMAGE_SIZE, MODEL_PATH, NUM_CLASSES, OUTPUT_DIR
from model import get_model, load_model_weights

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cityscapes Semantic Segmentation",
    page_icon="🚗",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a cleaner layout
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin: 2px 8px 2px 0;
        font-size: 13px;
    }
    .legend-swatch {
        width: 14px;
        height: 14px;
        border-radius: 2px;
        margin-right: 4px;
        border: 1px solid #444;
        display: inline-block;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained DeepLabV3 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(pretrained=False).to(device)

    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        # Fallback: try relative to app directory
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "best_model.pth"
        )

    load_model_weights(model, model_path, device)
    model.eval()
    return model, device


@st.cache_data
def load_evaluation_results():
    """Load pre-computed evaluation results if available."""
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    if not os.path.exists(results_path):
        # Fallback
        results_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", "evaluation_results.json"
        )
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            return json.load(f)
    return None


def run_inference(model, image: Image.Image, device) -> np.ndarray:
    """Run segmentation on a single image.

    Args:
        model:  Loaded PyTorch model.
        image:  PIL Image (RGB).
        device: torch.device.

    Returns:
        H×W numpy array of predicted class labels (0–18).
    """
    resized = image.resize(IMAGE_SIZE, Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)["out"]
        prediction = output.argmax(1).squeeze(0).cpu().numpy()

    return prediction


def render_legend():
    """Render the color legend as HTML."""
    legend_html = '<div style="line-height: 1.8;">'

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CITYSCAPES_COLORMAP)):
        r, g, b = color
        legend_html += (
            f'<span class="legend-item">'
            f'<span class="legend-swatch" style="background:rgb({r},{g},{b})"></span>'
            f'{name}'
            f'</span>'
        )

    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # --- Header ----------------------------------------------------------
    st.title("🚗 Semantic Scene Parsing")
    st.markdown(
        "Upload a street image to see per-pixel semantic segmentation "
        "using **DeepLabV3-ResNet50** fine-tuned on Cityscapes."
    )

    # --- Load model ------------------------------------------------------
    try:
        model, device = load_model()
        st.sidebar.success(f"Model loaded on **{device}**")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # --- Sidebar controls ------------------------------------------------
    st.sidebar.header("Controls")

    alpha = st.sidebar.slider(
        "Overlay opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 = original image only, 1 = mask only",
    )

    show_legend = st.sidebar.checkbox("Show class legend", value=True)
    show_mask_only = st.sidebar.checkbox("Show mask without overlay", value=False)

    # --- Sidebar: evaluation results -------------------------------------
    eval_results = load_evaluation_results()
    if eval_results:
        st.sidebar.header("Model Performance")
        st.sidebar.metric("Pixel Accuracy", f"{eval_results['pixel_accuracy']:.4f}")
        st.sidebar.metric("Mean IoU", f"{eval_results['miou']:.4f}")

        with st.sidebar.expander("Per-class IoU"):
            for class_name, iou in eval_results["per_class_iou"].items():
                st.text(f"{class_name:<16} {iou:.4f}")

    # --- File upload -----------------------------------------------------
    uploaded_file = st.file_uploader(
        "Choose a street image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload any street/urban image for semantic segmentation",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # --- Run inference -----------------------------------------------
        with st.spinner("Running segmentation..."):
            prediction = run_inference(model, image, device)

        mask_rgb = colorize_mask(prediction)
        resized_image = np.array(image.resize(IMAGE_SIZE, Image.BILINEAR))
        blended = overlay_mask(resized_image, mask_rgb, alpha=alpha)

        # --- Display results ---------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            if show_mask_only:
                st.subheader("Segmentation Mask")
                st.image(mask_rgb, use_container_width=True)
            else:
                st.subheader("Segmentation Overlay")
                st.image(blended, use_container_width=True)

        # --- Legend ------------------------------------------------------
        if show_legend:
            st.subheader("Class Legend")
            render_legend()

        # --- Prediction stats --------------------------------------------
        with st.expander("Prediction Statistics"):
            unique, counts = np.unique(prediction, return_counts=True)
            total_pixels = prediction.size

            stats_data = []
            for cls_id, count in zip(unique, counts):
                if cls_id < len(CLASS_NAMES):
                    name = CLASS_NAMES[cls_id]
                    pct = count / total_pixels * 100
                    stats_data.append({
                        "Class": name,
                        "Pixels": int(count),
                        "Percentage": f"{pct:.1f}%",
                    })

            if stats_data:
                st.table(stats_data)

    else:
        # --- Placeholder -------------------------------------------------
        st.info(
            "👆 Upload a street image above to get started.\n\n"
            "The model will assign one of 19 semantic classes to every pixel: "
            "road, sidewalk, building, car, person, vegetation, sky, and more."
        )


if __name__ == "__main__":
    main()