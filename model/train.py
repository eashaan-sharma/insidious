import torch
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=3
)

model.to(device)

print("Swin Transformer loaded successfully on", device)

