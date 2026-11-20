import glob
import os
import cv2
import numpy as np


def generate_synthetic_checkerboards(
    output_dir: str,
    pattern_size: tuple[int, int] = (9, 6),
    square_size_px: int = 40,
    num_views: int = 12,
    img_size: tuple[int, int] = (640, 480),
) -> list[str]:
    """Generates synthetic distorted checkerboard views using a known pinhole camera model."""
    os.makedirs(output_dir, exist_ok=True)
    w, h = img_size
    cols, rows = pattern_size

    # Create planar checkerboard image with white border
    board_w = (cols + 3) * square_size_px
    board_h = (rows + 3) * square_size_px
    board = np.ones((board_h, board_w), dtype=np.uint8) * 255
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = (r + 1) * square_size_px
                x0 = (c + 1) * square_size_px
                board[y0 : y0 + square_size_px, x0 : x0 + square_size_px] = 0

    # Ground-truth pinhole intrinsics and radial/tangential distortion
    fx = fy = 520.0
    cx, cy = w / 2.0, h / 2.0
    k_true = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_true = np.array([-0.28, 0.11, 0.001, -0.001, 0.0], dtype=np.float64)

    rng = np.random.default_rng(42)
    saved_paths = []
    src_pts = np.array(
        [[0, 0], [board_w - 1, 0], [board_w - 1, board_h - 1], [0, board_h - 1]],
        dtype=np.float32,
    )

    for i in range(num_views):
        margin_x = rng.uniform(40, 110, size=(4,))
        margin_y = rng.uniform(30, 80, size=(4,))
        dst_pts = np.array(
            [
                [margin_x[0], margin_y[0]],
                [w - margin_x[1], margin_y[1] + rng.uniform(-15, 15)],
                [w - margin_x[2], h - margin_y[2]],
                [margin_x[3], h - margin_y[3] + rng.uniform(-15, 15)],
            ],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src_pts, dst_pts)
        warped = cv2.warpPerspective(board, H, (w, h), borderValue=200)

        # Apply lens distortion via inverse mapping
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).reshape(-1, 1, 2)
        undist_norm = cv2.undistortPoints(pts, k_true, None)
        dist_pts, _ = cv2.projectPoints(
            np.concatenate([undist_norm[:, 0, :], np.ones((len(pts), 1))], axis=1),
            np.zeros(3),
            np.zeros(3),
            k_true,
            dist_true,
        )
        map_x = dist_pts[:, 0, 0].reshape(h, w).astype(np.float32)
        map_y = dist_pts[:, 0, 1].reshape(h, w).astype(np.float32)
        distorted = cv2.remap(warped, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=200)
        distorted_bgr = cv2.cvtColor(distorted, cv2.COLOR_GRAY2BGR)

        path = os.path.join(output_dir, f"calib_view_{i:02d}.png")
        cv2.imwrite(path, distorted_bgr)
        saved_paths.append(path)

    return saved_paths


def calibrate_camera_from_folder(
    calib_dir: str = "data/calibration",
    pattern_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
) -> dict:
    """Calibrates camera intrinsics K and distortion coefficients from checkerboard images."""
    image_paths = sorted(
        glob.glob(os.path.join(calib_dir, "*.png"))
        + glob.glob(os.path.join(calib_dir, "*.jpg"))
        + glob.glob(os.path.join(calib_dir, "*.jpeg"))
    )
    if len(image_paths) < 5:
        image_paths = generate_synthetic_checkerboards(calib_dir, pattern_size=pattern_size)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size_mm

    obj_points, img_points = [], []
    img_shape = None
    sample_detected = None
    sample_raw = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_sub)
            if sample_detected is None:
                sample_raw = img.copy()
                sample_detected = cv2.drawChessboardCorners(img.copy(), pattern_size, corners_sub, found)

    if len(obj_points) == 0:
        raise ValueError("No valid checkerboard patterns detected in calibration images.")

    rms_error, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )

    # Undistort sample checkerboard
    h, w = sample_raw.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(sample_raw, K, dist_coeffs, None, new_K)

    return {
        "num_valid_views": len(obj_points),
        "rms_reprojection_error": float(rms_error),
        "camera_matrix_K": K,
        "distortion_coefficients": dist_coeffs,
        "sample_raw": sample_raw,
        "sample_corners": sample_detected,
        "sample_undistorted": undistorted,
    }


def undistort_image(image_bgr: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Applies calibrated pinhole undistortion to any input image."""
    h, w = image_bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    return cv2.undistort(image_bgr, K, dist_coeffs, None, new_K)
