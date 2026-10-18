import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)


# Choose the best device available on the host. Railway CPU deployments use
# the final fallback without requiring any CUDA or Apple-specific packages.
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# Load Depth Anything V2
model_name = "depth-anything/Depth-Anything-V2-Small-hf"

processor = None
model = None


def estimate_depth(image_path):
    """
    Estimate relative depth for an image.

    Returns:
        NumPy depth map with the same spatial dimensions
        as the original image.
    """

    global processor, model
    if model is None:
        processor = AutoImageProcessor.from_pretrained(model_name)
        loaded = AutoModelForDepthEstimation.from_pretrained(model_name)
        loaded.to(device)
        loaded.eval()
        model = loaded

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare input
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Move tensors to device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Run depth model
    with torch.no_grad():
        outputs = model(**inputs)

    # Predicted depth
    predicted_depth = outputs.predicted_depth

    # Original image dimensions
    original_width, original_height = image.size

    # Resize depth map to original image size
    resized_depth = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False
    )

    # Convert to NumPy
    depth = resized_depth.squeeze().cpu().numpy()

    return depth
