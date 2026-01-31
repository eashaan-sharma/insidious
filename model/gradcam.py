import torch
import cv2
import numpy as np
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam[0].cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam


def generate_gradcam(model, input_tensor, save_path="static/heatmaps"):
    os.makedirs(save_path, exist_ok=True)

    cam_generator = GradCAM(model, model.layer4)
    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()

    cam = cam_generator.generate(input_tensor, class_idx)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_path = os.path.join(save_path, "cam.png")
    cv2.imwrite(heatmap_path, heatmap)

    return heatmap_path
