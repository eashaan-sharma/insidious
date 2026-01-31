import torch
import torch.nn.functional as F
from model_def import get_model
from dataset import get_dataloader
from inference import load_model
from gradcam import generate_gradcam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(weights_path, save_cams=False):
    # Load model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Validation data
    val_loader = get_dataloader(
        data_dir="../chest_xray/val",
        batch_size=1,
        train=False
    )

    TP = FP = TN = FN = 0
    confidences = []

    with torch.no_grad():
        for idx, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, dim=1)

            confidences.append(confidence.item())

            # Confusion matrix
            if label.item() == 1 and pred.item() == 1:
                TP += 1
            elif label.item() == 0 and pred.item() == 1:
                FP += 1
            elif label.item() == 0 and pred.item() == 0:
                TN += 1
            elif label.item() == 1 and pred.item() == 0:
                FN += 1

            # Optional: save Grad-CAM for interesting cases
            if save_cams and idx < 10:
                generate_gradcam(model, image, save_path=f"static/heatmaps/sample_{idx}.png")

    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)

    print("Evaluation Results")
    print("------------------")
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall (Pneumonia): {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "mean_confidence": sum(confidences) / len(confidences)
    }


if __name__ == "__main__":
    evaluate_model("cnn_weights.pth", save_cams=True)
