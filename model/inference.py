import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from model.model_def import get_model
from model.gradcam import (
    generate_gradcam,
    overlay_heatmap_on_image,
    localize_from_heatmap
)
import uuid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(weights_path=None):
    model = get_model()
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model
def predict(image_tensor,model):
    """
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('model\experiments\resnet_uncertainty_3epochs.pth', map_location=device))
    model.to(device)"""
    
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    unique_id = str(uuid.uuid4())

    heatmap_filename = f"cam_{unique_id}.png"
    heatmap_path = generate_gradcam(
        model,
        image_tensor,
        filename=heatmap_filename
    )



    risk_flag = (
        "high_risk" if confidence.item() > 0.8
        else "review_recommended"
    )

    # --------------------
    # NEW: overlay + boxes
    # --------------------
    overlay_path = None
    boxes = None

    try:
        pil_image = transforms.ToPILImage()(image_tensor.squeeze().cpu())
        overlay = overlay_heatmap_on_image(pil_image, heatmap_path)

        overlay_filename = f"overlay_{unique_id}.png"
        overlay_path = f"static/heatmaps/{overlay_filename}"
        cv2.imwrite(overlay_path, overlay)


        boxes = localize_from_heatmap(heatmap_path)

    except Exception as e:
        print("Explainability extension failed:", e)

    return {
        "prediction": int(pred.item()),
        "confidence": float(confidence.item()),
        "risk_flag": risk_flag,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "boxes": boxes
    }
