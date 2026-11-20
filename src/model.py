import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from src.dataset import IDX_TO_CLASS, get_transforms


class ResidualBlock(nn.Module):
    """3x3 Residual Convolutional Block with BatchNorm and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFAKEResNet(nn.Module):
    """Lightweight Residual CNN designed specifically for 32x32 CIFAKE authenticity classification."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock(32, 64, stride=1)    # 32x32
        self.stage2 = ResidualBlock(64, 128, stride=2)   # 16x16
        self.stage3 = ResidualBlock(128, 256, stride=2)  # 8x8
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x)


class GradCAM:
    """Computes Grad-CAM visual attention heatmaps on the final residual stage."""

    def __init__(self, model: CIFAKEResNet):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = self.model.stage3.conv2

        self._fwd_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = F.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def load_trained_model(weights_path: str = "models/cifake_cnn_best.pth", device: str | None = None):
    """Loads saved CIFAKEResNet checkpoint for inference."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFAKEResNet(num_classes=2).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, device


def predict_with_gradcam(image_bgr: np.ndarray, model: CIFAKEResNet, device: str) -> dict:
    """Runs authenticity prediction and generates a Grad-CAM overlay heatmap."""
    _, eval_tf = get_transforms(img_size=32)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    tensor_in = eval_tf(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_in)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])

    gradcam = GradCAM(model)
    try:
        cam_small = gradcam.generate_cam(tensor_in, pred_idx)
    finally:
        gradcam.remove_hooks()

    vis_size = (256, 256)
    base_vis = cv2.resize(image_bgr, vis_size, interpolation=cv2.INTER_CUBIC)
    cam_resized = cv2.resize(cam_small, vis_size, interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(base_vis, 0.55, heatmap_color, 0.45, 0)

    return {
        "predicted_class": pred_label,
        "confidence": confidence,
        "probabilities": {
            "FAKE": float(probs[0]),
            "REAL": float(probs[1]),
        },
        "gradcam_overlay_bgr": overlay_bgr,
        "gradcam_heatmap_bgr": heatmap_color,
    }
