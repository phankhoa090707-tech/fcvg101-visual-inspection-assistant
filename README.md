# FCVG101 Visual Inspection Assistant (CIFAKE Authenticity Inspector)

An end-to-end Computer Vision Visual Inspection Assistant developed for **FCVG101: Introduction to Computer Vision**, combining classical geometric, spatial, and frequency-domain image processing (**OpenCV + NumPy**) with a custom residual convolutional neural network and Grad-CAM explainability (**PyTorch**) inside an interactive **Streamlit** application.

---

## 1. Repository Structure

```text
fcvg101_cv_assistant/
├── data/
│   ├── calibration/          # Checkerboard views (auto-generated if empty)
│   └── cifake/               # CIFAKE dataset (ignored in Git; see Section 3)
│       ├── train/
│       │   ├── REAL/
│       │   └── FAKE/
│       └── test/
│           ├── REAL/
│           └── FAKE/
├── models/
│   └── cifake_cnn_best.pth   # Pre-trained CIFAKEResNet checkpoint (96.80% test accuracy)
├── outputs/                  # Exported evaluation metrics, plots, and Grad-CAM overlays
├── src/
│   ├── __init__.py
│   ├── calibration.py        # Task 1.1: Pinhole camera calibration & radial undistortion
│   ├── filtering.py          # Task 1.2: Spatial denoising, Sobel/Canny edges & 2D FFT
│   ├── features.py           # Task 1.3: Harris, SIFT, ORB & Lowe's ratio-test matching
│   ├── dataset.py            # Task 2.1: CIFAKE ingestion, cleaning, augmentation & splits
│   └── model.py              # Task 2.2: CIFAKEResNet architecture & Grad-CAM hooks
├── test_task1.py             # Runs Task 1 classical CV pipeline & saves visualizations
├── train_task2.py            # Runs Task 2 training, evaluation & exports model artifact
├── app.py                    # Task 3: Streamlit Visual Inspection Assistant GUI
└── requirements.txt
```

---

## 2. Environment Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/phankhoa090707-tech/fcvg101-visual-inspection-assistant.git](https://github.com/phankhoa090707-tech/fcvg101-visual-inspection-assistant.git)
   cd fcvg101-visual-inspection-assistant
   ```

2. **Create and activate a Python virtual environment (Windows PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install required dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

---

## 3. Dataset Setup (CIFAKE)

Because the CIFAKE dataset contains 120,000 images, `data/cifake/` is excluded via `.gitignore`.
1. Download the **CIFAKE: Real and AI-Generated Synthetic Images** dataset (100,000 train / 20,000 test RGB images at $32 \times 32$ resolution). link: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/code
2. Place the extracted folders into `data/cifake/` so the directory matches:
   - `data/cifake/train/REAL/` and `data/cifake/train/FAKE/`
   - `data/cifake/test/REAL/` and `data/cifake/test/FAKE/`

---

## 4. Running the Pipeline

### Task 1: Classical Image Analysis Engine (`test_task1.py`)
Runs pinhole camera calibration (12 checkerboard views, RMS reprojection error `2.2215 px`), spatial denoising benchmarks (Box, Gaussian, Median PSNR), Sobel/Canny edge gradients, 2D FFT spectral decomposition, and Harris/SIFT/ORB two-view feature matching:
```powershell
python test_task1.py
```

### Task 2: Deep Learning Model Training & Evaluation (`train_task2.py`)
A pre-trained checkpoint (`models/cifake_cnn_best.pth`) is already included in this repository. To retrain from scratch and regenerate `outputs/task2_metrics.json`, training curves, confusion matrix, and Grad-CAM overlays:
- **Full dataset (10 epochs, batch size 128):**
  ```powershell
  python train_task2.py --epochs 10 --batch-size 128
  ```
- **Fast CPU subset (20,000 training samples):**
  ```powershell
  python train_task2.py --epochs 8 --batch-size 128 --max-train-samples 20000
  ```

### Task 3: Visual Inspection Assistant Web App (`app.py`)
Launch the interactive Streamlit dashboard with the sidebar **Inspection Mode Router** (Full Inspection, Camera Undistortion, Spatial Filtering & 2D FFT, SIFT/ORB Matching, and Deep Learning + Grad-CAM):
```powershell
streamlit run app.py
```

---

## 5. Benchmark Results Summary

| Component | Metric / Evaluation | Result |
| :--- | :--- | :--- |
| **Task 1.1: Camera Calibration** | Valid Checkerboard Views / RMS Error | `12 views` / `2.2215 px` |
| **Task 1.2: Spatial Denoising** | Median Filter PSNR ($640\times480$ / $32\times32$) | `19.71 dB` / `38.02 dB` |
| **Task 1.3: Feature Matching** | SIFT Good Matches vs. ORB Good Matches | `304` (`35.6%`) vs. `92` (`18.4%`) |
| **Task 2: CIFAKEResNet (Test Set)** | Accuracy / Precision / Recall / F1-Score | `96.80%` / `97.57%` / `96.00%` / `0.9678` |
```