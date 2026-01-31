import torch
from model.model_def import get_model
from model.uncertainty import assign_risk
from model.gradcam import generate_gradcam


def load_model(weights_path=None):
    model = get_model()
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def predict(image_tensor, model):
    """
    Returns prediction, confidence, risk flag, heatmap
    """
    image_tensor = image_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    risk_flag = assign_risk(confidence.item())
    heatmap_path = generate_gradcam(model, image_tensor)


    return {
        "prediction": int(pred_class.item()),
        "confidence": float(confidence.item()),
        "risk_flag": risk_flag,
        "heatmap_path": heatmap_path
    }
if __name__ == "__main__":
    import torch
    from dataset import get_dataloader

    model = load_model("cnn_weights.pth")

    test_loader = get_dataloader(
        data_dir="../chest_xray/val",
        batch_size=1,
        train=False
    )

    for i, (image, label) in enumerate(test_loader):
        if i == 5:
            break

        result = predict(image, model)
        print("GT:", label.item(), "Pred:", result)

