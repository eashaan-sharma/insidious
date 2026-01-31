import torch
import torch.nn as nn
from tqdm import tqdm
import time
from model_def import get_model
from dataset import get_dataloader

# -----------------------------
# Device configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Train for one epoch
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    start_time = time.time()

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_time = time.time() - start_time
    print(f"Epoch completed in {epoch_time:.2f} seconds")

    return running_loss / len(dataloader)


# -----------------------------
# Main training function
# -----------------------------
def train():
    # Load CNN model (same as inference)
    model = get_model(num_classes=2)
    model = model.to(device)

    # DataLoader
    train_loader = get_dataloader(
        data_dir="../chest_xray/train",
        batch_size=32,
        train=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train for ONE epoch (enough for Eval 1)
    loss = train_one_epoch(model, train_loader, optimizer, criterion)

    print(f"Training loss: {loss:.4f}")

    # Save weights (optional but good)
    torch.save(model.state_dict(), "cnn_weights.pth")
    print("Model weights saved to cnn_weights.pth")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    train()
