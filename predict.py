import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from model.unet import UNet

DEFAULT_IMAGE_SIZE = 256
DEFAULT_THRESHOLD = 0.5


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[UNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def preprocess_image(image_path: str, image_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    original_pil = Image.open(image_path).convert("RGB")
    original_np = np.array(original_pil)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(original_pil).unsqueeze(0)
    return tensor, original_np


def sigmoid_confidence(logits: torch.Tensor) -> Tuple[np.ndarray, float]:
    probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    confidence = float(probs.max()) if probs.size > 0 else 0.0
    return probs, confidence


def postprocess_mask(prob_map: np.ndarray, original_shape: Tuple[int, int], threshold: float) -> np.ndarray:
    binary_mask = (prob_map >= threshold).astype(np.uint8) * 255
    resized_mask = cv2.resize(
        binary_mask,
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized_mask


def create_overlay(original_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = original_rgb.copy()
    color_mask = np.zeros_like(original_rgb)
    color_mask[:, :, 0] = mask
    color_mask[:, :, 1] = (mask > 0).astype(np.uint8) * 60
    color_mask[:, :, 2] = 0

    blended = cv2.addWeighted(overlay, 0.75, color_mask, 0.4, 0)
    return blended


def save_results(
    original_rgb: np.ndarray,
    mask: np.ndarray,
    overlay_rgb: np.ndarray,
    output_dir: str,
    sample_id: str,
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    original_file = output_path / f"{sample_id}_original.png"
    mask_file = output_path / f"{sample_id}_mask.png"
    overlay_file = output_path / f"{sample_id}_overlay.png"

    Image.fromarray(original_rgb).save(original_file)
    Image.fromarray(mask).save(mask_file)
    Image.fromarray(overlay_rgb).save(overlay_file)

    return {
        "original_image": str(original_file),
        "mask_image": str(mask_file),
        "overlay_image": str(overlay_file),
    }


@torch.no_grad()
def run_inference(
    image_path: str,
    checkpoint_path: str,
    output_dir: str = "outputs",
    sample_id: str = "prediction",
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint_path, device)
    image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))

    input_tensor, original_rgb = preprocess_image(image_path, image_size=image_size)
    input_tensor = input_tensor.to(device)

    logits = model(input_tensor)
    prob_map, confidence = sigmoid_confidence(logits)
    mask = postprocess_mask(prob_map, original_rgb.shape[:2], threshold)
    overlay = create_overlay(original_rgb, mask)

    saved = save_results(original_rgb, mask, overlay, output_dir, sample_id)
    coverage_percent = float((mask > 0).mean() * 100.0)

    return {
        **saved,
        "confidence": confidence,
        "coverage_percent": coverage_percent,
        "threshold": threshold,
    }


def cli():
    parser = argparse.ArgumentParser(description="Run inference using a trained U-Net model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the MRI image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    sample_id = Path(args.image).stem
    result = run_inference(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        sample_id=sample_id,
        threshold=args.threshold,
    )

    print("Inference complete.")
    print(f"Original: {result['original_image']}")
    print(f"Mask: {result['mask_image']}")
    print(f"Overlay: {result['overlay_image']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Coverage (%): {result['coverage_percent']:.2f}")


if __name__ == "__main__":
    cli()