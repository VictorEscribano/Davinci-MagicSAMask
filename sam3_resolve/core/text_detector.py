"""
Text-to-bounding-box detector using OWL-ViT (HuggingFace transformers).

Usage:
    from sam3_resolve.core.text_detector import detect

    box, score = detect(frame_bgr, "person")
    # box = [x1, y1, x2, y2] in pixel coords
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_processor = None
_model = None
_device: Optional[str] = None


def _load() -> tuple:
    global _processor, _model, _device
    if _model is not None:
        return _processor, _model, _device

    try:
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
    except ImportError:
        raise RuntimeError(
            "Text detection requires 'transformers'.\n"
            "Install with:  pip install transformers"
        )

    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading OWL-ViT checkpoint (first use — may download ~580 MB)…")
    _processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    _model = (
        OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        .to(_device)
        .eval()
    )
    logger.info("OWL-ViT ready on %s", _device)
    return _processor, _model, _device


def detect(
    frame_bgr: np.ndarray,
    query: str,
    threshold: float = 0.1,
) -> Optional[tuple[list[float], float]]:
    """
    Detect the object described by *query* in *frame_bgr*.

    Args:
        frame_bgr:  BGR numpy array (H, W, 3) from OpenCV.
        query:      Free-text description, e.g. "person" or "red car".
        threshold:  Minimum confidence to consider a detection valid.

    Returns:
        ([x1, y1, x2, y2], confidence) in pixel coordinates, or None if
        nothing was detected above *threshold*.
    """
    from PIL import Image
    import torch

    processor, model, device = _load()

    rgb = frame_bgr[..., ::-1].copy().astype(np.uint8)
    image = Image.fromarray(rgb)
    h, w = frame_bgr.shape[:2]

    # OWL-ViT expects a list-of-lists for multi-image batches
    texts = [[f"a photo of {query}"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]])
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )[0]

    if len(results["scores"]) == 0:
        return None

    best = int(results["scores"].argmax())
    score = float(results["scores"][best])
    box = [float(v) for v in results["boxes"][best].tolist()]
    return box, score
