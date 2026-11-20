import cv2
import numpy as np


def compute_psnr(clean: np.ndarray, restored: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (dB)."""
    mse = np.mean((clean.astype(np.float64) - restored.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def run_filtering_and_frequency_analysis(image_bgr: np.ndarray, kernel_size: int = 5) -> dict:
    """Runs spatial filtering, denoising evaluation, edge gradients, and 2D FFT analysis."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Synthetic Gaussian + Salt-and-Pepper noise injection for denoising benchmark
    rng = np.random.default_rng(42)
    noisy = gray.astype(np.float32) + rng.normal(0, 18.0, gray.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    sp_mask = rng.random(gray.shape)
    noisy[sp_mask < 0.015] = 0
    noisy[sp_mask > 0.985] = 255

    # Spatial Filters
    box_filtered = cv2.blur(noisy, (kernel_size, kernel_size))
    gaussian_filtered = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), sigmaX=1.2)
    median_filtered = cv2.medianBlur(noisy, kernel_size)

    psnr_metrics = {
        "noisy_psnr_db": round(compute_psnr(gray, noisy), 2),
        "box_psnr_db": round(compute_psnr(gray, box_filtered), 2),
        "gaussian_psnr_db": round(compute_psnr(gray, gaussian_filtered), 2),
        "median_psnr_db": round(compute_psnr(gray, median_filtered), 2),
    }

    # 2. Image Gradients (Sobel & Canny)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag_u8 = np.uint8(255 * sobel_mag / (np.max(sobel_mag) + 1e-8))
    canny_edges = cv2.Canny(gray, threshold1=80, threshold2=180)

    # 3. 2D Frequency-Domain Analysis (FFT)
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1.0)
    spectrum_vis = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # High-pass & Low-pass circular masks
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    y_grid, x_grid = np.ogrid[:rows, :cols]
    radius = max(4, min(rows, cols) // 6)
    dist_sq = (y_grid - crow) ** 2 + (x_grid - ccol) ** 2

    low_pass_mask = (dist_sq <= radius**2).astype(np.float32)
    high_pass_mask = 1.0 - low_pass_mask

    img_low_pass = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * low_pass_mask)))
    img_high_pass = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * high_pass_mask)))

    img_low_pass_u8 = cv2.normalize(img_low_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_high_pass_u8 = cv2.normalize(img_high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    high_freq_energy_ratio = float(
        np.sum(np.abs(f_shift) * high_pass_mask) / (np.sum(np.abs(f_shift)) + 1e-8)
    )

    return {
        "gray": gray,
        "noisy": noisy,
        "box_filtered": box_filtered,
        "gaussian_filtered": gaussian_filtered,
        "median_filtered": median_filtered,
        "sobel_magnitude": sobel_mag_u8,
        "canny_edges": canny_edges,
        "fft_spectrum": spectrum_vis,
        "fft_low_pass": img_low_pass_u8,
        "fft_high_pass": img_high_pass_u8,
        "metrics": {
            **psnr_metrics,
            "mean_gradient_magnitude": round(float(np.mean(sobel_mag)), 2),
            "edge_pixel_ratio": round(float(np.mean(canny_edges > 0)), 4),
            "high_freq_energy_ratio": round(high_freq_energy_ratio, 4),
        },
    }
