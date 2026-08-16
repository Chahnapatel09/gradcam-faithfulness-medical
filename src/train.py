import os

import pandas as pd
import torch
import torch.nn as nn

from data import get_dataloaders
from model import build_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.squeeze().long().to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(
    data_dir: str = "./data",
    checkpoint_dir: str = "./checkpoints",
    results_dir: str = "./results",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(data_dir=data_dir, batch_size=batch_size)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    best_path = os.path.join(checkpoint_dir, "best_model.pth")
    log_path = os.path.join(results_dir, "training_log.csv")

    start_epoch = 0
    best_val_acc = 0.0
    history = []

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        if os.path.exists(log_path):
            history = pd.read_csv(log_path).to_dict("records")
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        pd.DataFrame(history).to_csv(log_path, index=False)

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }, checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  New best val_acc={val_acc:.4f}, saved to {best_path}")

    return model


if __name__ == "__main__":
    train()
