import time
from typing import Any, Dict, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..utils import ExperimentTracker, setup_logger

logger = setup_logger("Trainer")
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # TODO: Define Loss Function (Criterion)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: Initialize ExperimentTracker
        self.train_losses = []
        self.val_losses = []
        
        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # TODO: Implement Training Loop
        # 1. Iterate over dataloader
        # 2. Move data to device
        # 3. Forward pass, Calculate Loss
        # 4. Backward pass, Optimizer step
        # 5. Track metrics (Loss, Accuracy, F1)
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch_idx+1} [Train]"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        logger.info(f"Epoch {epoch_idx+1} Train loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss

    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        
        # TODO: Implement Validation Loop
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch_idx+1} [Val]"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        logger.info(f"Epoch {epoch_idx+1} Val loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        save_dir = Path(self.config["training"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, save_dir / f"checkpoint_epoch_{epoch}.pt")

    def save_plot(self) -> None:
        save_dir = Path(self.config["training"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10,5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(save_dir / "loss_curve.png")
        plt.close()
        logger.info(f"Saved loss curce to {save_dir / 'loss_curve.png'}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        
        print(f"Starting training for {epochs} epochs...")
        logger.info(f"Startin training for {epochs} epochs...")
        for epoch in range(epochs):
            # TODO: Call train_epoch and validate
            # TODO: Log metrics to tracker
            # TODO: Save checkpoints
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.save_checkpoint(epoch, val_loss)
        self.save_plot()
        logger.info("Training complete.")
            
	# Remember to handle the trackers properly
