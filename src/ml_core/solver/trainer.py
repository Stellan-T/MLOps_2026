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
        self.log_every = config["training"].get("log_every", 50)
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
            if (batch_idx + 1) % self.log_every ==0:    
                avg_loss = running_loss / self.log_every
                self.train_losses.append(avg_loss)
                logger.info(f"Epoch {epoch_idx+1} Step: {batch_idx+1}, Loss: {avg_loss:.4f}")
                running_loss = 0.0
        accuracy = 100.0 * correct / total
        logger.info(f"Epoch {epoch_idx+1} Train Accuracy: {accuracy:.2f}%")
        return accuracy

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(range(1, len(self.train_losses) + 1), self.train_losses)
        ax1.set_xlabel(f"Step (every {self.log_every} batches)")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss (per step)")
        ax2.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'o-')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Loss (per epoch)")
        plt.tight_layout()
        plt.savefig(save_dir / "loss_curve.png")
        plt.close()
        logger.info(f"Saved loss curve to {save_dir / 'loss_curve.png'}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        
        print(f"Starting training for {epochs} epochs...")
        logger.info(f"Startin training for {epochs} epochs...")
        for epoch in range(epochs):
            # TODO: Call train_epoch and validate
            # TODO: Log metrics to tracker
            # TODO: Save checkpoints
            self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            self.save_checkpoint(epoch, val_loss)
        self.save_plot()
        logger.info("Training complete.")
            
	# Remember to handle the trackers properly
