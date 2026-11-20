import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.calibration import calibrate_camera_from_folder, undistort_image
from src.features import detect_and_match_features
from src.filtering import run_filtering_and_frequency_analysis
from src.model import load_trained_model, predict_with_gradcam

st.set_page_config(page_title="FCVG101 Visual Inspection Assistant", layout="wide")


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_uploaded_bgr(uploaded_file) -> np.ndarray:
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@st.cache_resource
def get_calibration_state():
    return calibrate_camera_from_folder("data/calibration")


@st.cache_resource
def get_dl_model(weights_path: str):
    if not os.path.exists(weights_path):
        return None, "cpu"
    return load_trained_model(weights_path)


st.title("Visual Inspection Assistant (FCVG101 MVP)")
st.caption("Digital Media & Catalog Authenticity Inspector — Classical CV Pipeline + PyTorch Residual CNN")

st.sidebar.header("Inspection Mode (Tool Router)")
mode = st.sidebar.selectbox(
    "Select Assistant Task:",
    [
        "Full Inspection (All Modules)",
        "1. Camera Calibration & Undistortion",
        "2. Spatial Filtering, Edges & 2D FFT",
        "3. Keypoint Detection & Two-View Matching",
        "4. Deep Learning Authenticity Classification (Grad-CAM)",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Pipeline Parameters")
kernel_size = st.sidebar.slider("Spatial Filter Kernel Size", min_value=3, max_value=9, value=3, step=2)
ratio_thresh = st.sidebar.slider("Lowe's Ratio Test Threshold", min_value=0.50, max_value=0.90, value=0.75, step=0.05)
weights_path = st.sidebar.text_input("Model Weights Path", value="models/cifake_cnn_best.pth")

col_up1, col_up2 = st.columns(2)
with col_up1:
    primary_file = st.file_uploader("Upload Primary Image to Inspect", type=["png", "jpg", "jpeg", "webp"])
with col_up2:
    second_file = st.file_uploader(
        "Optional Second View (for Two-View Feature Matching)", type=["png", "jpg", "jpeg", "webp"]
    )

calib_res = get_calibration_state()

if primary_file is None:
    st.info("Upload an image above (e.g., from `data/cifake/test/REAL` or `FAKE`) to run the visual inspection pipeline.")
    with st.expander("Preview Active Camera Calibration Parameters (Task 1.1)"):
        c1, c2 = st.columns(2)
        c1.metric("Valid Calibration Views", calib_res["num_valid_views"])
        c2.metric("RMS Reprojection Error", f"{calib_res['rms_reprojection_error']:.4f} px")
        st.write("**Intrinsic Matrix $K$:**")
        st.code(np.array2string(calib_res["camera_matrix_K"], precision=3))
        st.write("**Distortion Coefficients ($k_1, k_2, p_1, p_2, k_3$):**")
        st.code(np.array2string(calib_res["distortion_coefficients"], precision=4))
    st.stop()

img_bgr = load_uploaded_bgr(primary_file)
img_undist = undistort_image(img_bgr, calib_res["camera_matrix_K"], calib_res["distortion_coefficients"])
img2_bgr = load_uploaded_bgr(second_file) if second_file is not None else None

if mode in ("Full Inspection (All Modules)", "1. Camera Calibration & Undistortion"):
    st.header("1. Camera Geometry & Pinhole Undistortion")
    m1, m2, m3 = st.columns(3)
    m1.metric("Calibration Views", calib_res["num_valid_views"])
    m2.metric("RMS Reprojection Error", f"{calib_res['rms_reprojection_error']:.4f} px")
    m3.metric("Input Resolution", f"{img_bgr.shape[1]} x {img_bgr.shape[0]} px")

    c1, c2, c3, c4 = st.columns(4)
    c1.image(bgr_to_rgb(calib_res["sample_corners"]), caption="Checkerboard Corners", use_container_width=True)
    c2.image(bgr_to_rgb(calib_res["sample_undistorted"]), caption="Undistorted Board", use_container_width=True)
    c3.image(bgr_to_rgb(img_bgr), caption="Raw Uploaded Image", use_container_width=True)
    c4.image(bgr_to_rgb(img_undist), caption="Undistorted Inspection Image", use_container_width=True)

    with st.expander("View Camera Matrix K and Distortion Coefficients"):
        st.code(f"K =\n{calib_res['camera_matrix_K']}\n\ndist = {calib_res['distortion_coefficients']}")

if mode in ("Full Inspection (All Modules)", "2. Spatial Filtering, Edges & 2D FFT"):
    st.header("2. Spatial Filtering, Gradients & 2D Frequency Analysis")
    filt_res = run_filtering_and_frequency_analysis(img_undist, kernel_size=kernel_size)
    metrics = filt_res["metrics"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Noisy PSNR", f"{metrics['noisy_psnr_db']} dB")
    k2.metric("Median Denoised PSNR", f"{metrics['median_psnr_db']} dB", delta=f"+{metrics['median_psnr_db'] - metrics['noisy_psnr_db']:.2f} dB")
    k3.metric("Mean Sobel Gradient", f"{metrics['mean_gradient_magnitude']}")
    k4.metric("High-Freq Energy Ratio", f"{metrics['high_freq_energy_ratio']:.4f}")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.image(filt_res["noisy"], caption="Injected Noise", use_container_width=True)
    r1c2.image(filt_res["box_filtered"], caption=f"Box Filter (PSNR {metrics['box_psnr_db']} dB)", use_container_width=True)
    r1c3.image(filt_res["gaussian_filtered"], caption=f"Gaussian Filter (PSNR {metrics['gaussian_psnr_db']} dB)", use_container_width=True)
    r1c4.image(filt_res["median_filtered"], caption=f"Median Filter (PSNR {metrics['median_psnr_db']} dB)", use_container_width=True)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.image(filt_res["sobel_magnitude"], caption="Sobel Gradient Magnitude", use_container_width=True)
    r2c2.image(filt_res["canny_edges"], caption="Canny Edges", use_container_width=True)
    r2c3.image(filt_res["fft_spectrum"], caption="2D FFT Magnitude Spectrum", use_container_width=True)
    r2c4.image(filt_res["fft_high_pass"], caption="High-Pass Filtered Structure", use_container_width=True)

if mode in ("Full Inspection (All Modules)", "3. Keypoint Detection & Two-View Matching"):
    st.header("3. Interest-Point Detection & Two-View Matching")
    feat_res = detect_and_match_features(img_undist, img2_bgr=img2_bgr, ratio_thresh=ratio_thresh)
    fmetrics = feat_res["metrics"]

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Harris Corner Pixels", fmetrics["harris_corner_pixels"])
    fc2.metric("SIFT Good Matches (Ratio Test)", f"{fmetrics['sift_good_matches']} / {fmetrics['sift_keypoints_view1']} kps")
    fc3.metric("ORB Good Matches (Ratio Test)", f"{fmetrics['orb_good_matches']} / {fmetrics['orb_keypoints_view1']} kps")

    v1, v2, v3 = st.columns(3)
    v1.image(bgr_to_rgb(feat_res["harris_vis"]), caption="Harris Corners", use_container_width=True)
    v2.image(bgr_to_rgb(feat_res["sift_keypoints_vis"]), caption="SIFT Keypoints (Scale & Orientation)", use_container_width=True)
    v3.image(bgr_to_rgb(feat_res["orb_keypoints_vis"]), caption="ORB Keypoints", use_container_width=True)

    m_col1, m_col2 = st.columns(2)
    m_col1.image(bgr_to_rgb(feat_res["sift_matches_vis"]), caption="SIFT Two-View Matches (Lowe's Ratio Test)", use_container_width=True)
    m_col2.image(bgr_to_rgb(feat_res["orb_matches_vis"]), caption="ORB Two-View Matches (Hamming Ratio Test)", use_container_width=True)

if mode in ("Full Inspection (All Modules)", "4. Deep Learning Authenticity Classification (Grad-CAM)"):
    st.header("4. Deep Learning Authenticity Assessment (CIFAKEResNet + Grad-CAM)")
    model, device = get_dl_model(weights_path)
    if model is None:
        st.warning(
            f"Trained model weights not found at `{weights_path}`. Run `python train_task2.py` first to generate the model artifact."
        )
    else:
        dl_res = predict_with_gradcam(img_undist, model, device)
        pred_label = dl_res["predicted_class"]
        conf = dl_res["confidence"] * 100.0

        d1, d2, d3 = st.columns(3)
        d1.metric("Predicted Authenticity", pred_label)
        d2.metric("Model Confidence", f"{conf:.2f}%")
        d3.metric("Compute Device", device.upper())

        g1, g2, g3 = st.columns(3)
        g1.image(cv2.resize(bgr_to_rgb(img_undist), (256, 256), interpolation=cv2.INTER_NEAREST), caption="Inspected Image (256x256 View)", use_container_width=True)
        g2.image(bgr_to_rgb(dl_res["gradcam_heatmap_bgr"]), caption="Grad-CAM Spatial Attention Map", use_container_width=True)
        g3.image(bgr_to_rgb(dl_res["gradcam_overlay_bgr"]), caption=f"Grad-CAM Overlay ({pred_label})", use_container_width=True)

        st.bar_chart(dl_res["probabilities"])
