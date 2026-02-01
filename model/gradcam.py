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
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        input_tensor.requires_grad = True

        with torch.enable_grad():
            self.model.zero_grad()
            output = self.model(input_tensor)
            output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


def generate_gradcam(model, input_tensor, save_dir="static/heatmaps", filename="cam.png"):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()  # IMPORTANT

    cam_generator = GradCAM(model, model.layer4[-1])

    # forward + backward happens INSIDE generate()
    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()

    cam = cam_generator.generate(input_tensor, class_idx)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_path = os.path.join(save_dir, filename)
    cv2.imwrite(heatmap_path, heatmap)

    return heatmap_path
def overlay_heatmap_on_image(original_image, heatmap_path, alpha=0.4):
    img = np.array(original_image.resize((224,224)))[:,:,::-1]
    heatmap = cv2.imread(heatmap_path)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay


def localize_from_heatmap(heatmap_path, threshold=150):
    heatmap_color = cv2.imread(heatmap_path)
    heatmap_gray = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]
    return boxes

