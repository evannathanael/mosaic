"""Creator Balance — interactive demo dashboard.

Upload an image, optionally apply a transform live, and see:
  - AI-generation probability (calibrated)
  - Which of the 6 required transforms most affects the prediction (robustness view)
  - Near-duplicate cluster info (when running against a folder of images)

Run with:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src...`

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data.transforms import NAMED_EVAL_TRANSFORMS, named_eval_transform
from src.models.classifier import AIGCDetector
from src.utils import load_config

st.set_page_config(page_title="Creator Balance", layout="wide")


@st.cache_resource
def load_model(checkpoint_path: str, config_path: str):
    config = load_config(config_path)
    model = AIGCDetector.load(checkpoint_path, config, device="cpu")
    return model, config


def predict(model, config, pil_img: Image.Image) -> float:
    image_size = config["data"]["image_size"]
    img = pil_img.resize((image_size, image_size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        prob = model.predict_proba(tensor).item()
    return prob


st.title("🎨 Creator Balance")
st.caption(
    "AI-image detection + repetition tracking — flags high-volume repeated "
    "synthetic content without penalizing one-off, legitimate AI creativity."
)

with st.sidebar:
    st.header("Settings")
    checkpoint_path = st.text_input("Model checkpoint path", value="outputs/baseline/model_best.pt")
    config_path = st.text_input("Config path", value="configs/config.yaml")
    transform_name = st.selectbox("Simulate a transform (robustness demo)", list(NAMED_EVAL_TRANSFORMS.keys()))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        model, config = load_model(checkpoint_path, config_path)
    except FileNotFoundError:
        st.error(
            f"No checkpoint found at `{checkpoint_path}`. Train a model first with "
            "`python src/models/train.py --config configs/config.yaml`."
        )
        st.stop()

    original_img = Image.open(uploaded_file).convert("RGB")
    transformed_arr = named_eval_transform(transform_name, np.array(original_img))
    transformed_img = Image.fromarray(transformed_arr)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(original_img, use_container_width=True)
        prob_clean = predict(model, config, original_img)
        st.metric("AI-generated probability (clean)", f"{prob_clean:.1%}")

    with col2:
        st.subheader(f"After transform: {transform_name}")
        st.image(transformed_img, use_container_width=True)
        prob_transformed = predict(model, config, transformed_img)
        st.metric(
            "AI-generated probability (transformed)",
            f"{prob_transformed:.1%}",
            delta=f"{(prob_transformed - prob_clean):+.1%}",
        )

    st.divider()
    st.subheader("Content diversity signal")
    st.info(
        "Run `python src/inference.py --input_dir <folder>` on a full folder of "
        "images to see near-duplicate clustering and repetition scores across "
        "a batch — single-image similarity isn't meaningful on its own."
    )
else:
    st.info("Upload an image to get started.")
