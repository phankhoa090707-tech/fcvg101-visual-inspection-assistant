import cv2
import numpy as np


def _ensure_min_resolution(image_bgr: np.ndarray, min_dim: int = 256) -> np.ndarray:
    """Upscales tiny images (e.g., 32x32 CIFAKE) so local descriptors can extract keypoints."""
    h, w = image_bgr.shape[:2]
    if min(h, w) < min_dim:
        scale = min_dim / float(min(h, w))
        return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return image_bgr


def generate_second_view(image_bgr: np.ndarray, angle_deg: float = 18.0, scale: float = 0.9) -> np.ndarray:
    """Creates a geometric perspective/similarity second view for two-view matching."""
    h, w = image_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    return cv2.warpAffine(image_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def detect_and_match_features(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray | None = None,
    ratio_thresh: float = 0.75,
) -> dict:
    """Detects Harris, SIFT, and ORB keypoints and matches two views via Lowe's ratio test."""
    img1 = _ensure_min_resolution(img1_bgr)
    img2 = _ensure_min_resolution(img2_bgr) if img2_bgr is not None else generate_second_view(img1)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 1. Harris Corner Detection
    harris_resp = cv2.cornerHarris(np.float32(gray1), blockSize=2, ksize=3, k=0.04)
    harris_resp = cv2.dilate(harris_resp, None)
    harris_vis = img1.copy()
    harris_mask = harris_resp > 0.01 * harris_resp.max()
    harris_vis[harris_mask] = [0, 0, 255]
    harris_count = int(np.sum(harris_mask))

    # 2. SIFT Detection & KNN Ratio-Test Matching
    sift = cv2.SIFT_create()
    kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
    kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)

    sift_kp_vis = cv2.drawKeypoints(
        img1, kp1_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    good_sift = []
    if des1_sift is not None and des2_sift is not None and len(kp1_sift) >= 2 and len(kp2_sift) >= 2:
        bf_sift = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = bf_sift.knnMatch(des1_sift, des2_sift, k=2)
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio_thresh * n.distance:
                    good_sift.append(m)

    sift_match_vis = cv2.drawMatches(
        img1,
        kp1_sift,
        img2,
        kp2_sift,
        good_sift[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 3. ORB Detection & Hamming Ratio-Test Matching
    orb = cv2.ORB_create(nfeatures=500)
    kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
    kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)

    orb_kp_vis = cv2.drawKeypoints(img1, kp1_orb, None, color=(0, 255, 0))

    good_orb = []
    if des1_orb is not None and des2_orb is not None and len(kp1_orb) >= 2 and len(kp2_orb) >= 2:
        bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_orb = bf_orb.knnMatch(des1_orb, des2_orb, k=2)
        for m_pair in raw_orb:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio_thresh * n.distance:
                    good_orb.append(m)

    orb_match_vis = cv2.drawMatches(
        img1,
        kp1_orb,
        img2,
        kp2_orb,
        good_orb[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return {
        "harris_vis": harris_vis,
        "sift_keypoints_vis": sift_kp_vis,
        "sift_matches_vis": sift_match_vis,
        "orb_keypoints_vis": orb_kp_vis,
        "orb_matches_vis": orb_match_vis,
        "metrics": {
            "harris_corner_pixels": harris_count,
            "sift_keypoints_view1": len(kp1_sift),
            "sift_keypoints_view2": len(kp2_sift),
            "sift_good_matches": len(good_sift),
            "orb_keypoints_view1": len(kp1_orb),
            "orb_keypoints_view2": len(kp2_orb),
            "orb_good_matches": len(good_orb),
        },
    }
