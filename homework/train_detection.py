import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

from .models import Detector, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric
import os

# Define paths for different environments
if 'COLAB_GPU' in os.environ:  # Check if running on Google Colab
    base_path = '/content/drive/MyDrive/homeowrk3'
elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:  # Check if running on Kaggle
    base_path = '/kaggle/working/classification-detection'
else:  # Assume local machine
    base_path = '/Users/rituparna/Downloads/homework3'

# Define train and validation data paths
train_data_path = os.path.join(base_path, 'road_data/train')
val_data_path = os.path.join(base_path, 'road_data/val')


def dice_loss(pred, target, smooth=1.0):
    """
    Compute Dice loss between predicted and target tensors.
    Args:
        pred: Predicted tensor of shape [batch_size, num_classes, height, width].
        target: Ground truth tensor of shape [batch_size, height, width].
        smooth: Smoothing factor to avoid division by zero.
    """
    pred = torch.softmax(pred, dim=1)  # Apply softmax to get class probabilities
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()  # Return the Dice loss

def train_detection(
    exp_dir: str = 'logs',
    model_name: str = 'detection_model',
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 2024,
    depth_loss_weight: float = 2.0,
    dice_loss_weight: float = 2.0,
    **kwargs,
):
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a log directory with a timestamp
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the model and send it to the device
    model = Detector(**kwargs)
    model = model.to(device)

    # Load training and validation data
    train_data = load_data(train_data_path, transform_pipeline="aug", shuffle=True, batch_size=batch_size)
    val_data = load_data(val_data_path, shuffle=False)

    # Create loss functions and optimizer
    segmentation_loss_func = torch.nn.CrossEntropyLoss()
    depth_loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


    # Define metric for detection evaluation
    detection_metric = DetectionMetric(num_classes=3)

    global_step = 0

    # Training loop
    for epoch in range(num_epoch):
        # Reset metrics at the beginning of each epoch
        detection_metric.reset()

        model.train()

        for batch in train_data:
            img = batch['image'].to(device)
            labels = batch['track'].to(device)
            depth_labels = batch['depth'].to(device)

            # Forward pass
            logits, depth_preds = model(img)

            # Normalize depth predictions to the range [0, 1]
            depth_preds = torch.sigmoid(depth_preds.squeeze(1))

            # Compute loss
            segmentation_loss = segmentation_loss_func(logits, labels)
            dice_loss_value = dice_loss(logits, labels)
            depth_loss = depth_loss_func(depth_preds, depth_labels)

            # Combine losses with weights
            total_loss = segmentation_loss + dice_loss_weight * dice_loss_value + depth_loss_weight * depth_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update detection metric with training predictions
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            detection_metric.add(preds, labels, depth_preds, depth_labels)

            global_step += 1

        # Compute epoch training metrics
        epoch_train_metrics = detection_metric.compute()

        # Validation loop
        with torch.no_grad():
            model.eval()
            detection_metric.reset()  # Reset metrics for validation
            for batch in val_data:
                img = batch['image'].to(device)
                labels = batch['track'].to(device)
                depth_labels = batch['depth'].to(device)

                # Forward pass
                logits, depth_preds = model(img)
                depth_preds = torch.sigmoid(depth_preds.squeeze(1))  # Normalize depth predictions

                # Update detection metric with validation predictions
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                detection_metric.add(preds, labels, depth_preds, depth_labels)

        # Compute epoch validation metrics
        epoch_val_metrics = detection_metric.compute()

        # Print metrics
        epoch_train_acc = epoch_train_metrics["accuracy"]
        epoch_val_acc = epoch_val_metrics["accuracy"]
        epoch_train_iou = epoch_train_metrics["iou"]
        epoch_val_iou = epoch_val_metrics["iou"]
        epoch_train_mae = epoch_train_metrics["abs_depth_error"]
        epoch_val_mae = epoch_val_metrics["abs_depth_error"]
        epoch_train_tp_mae = epoch_train_metrics["tp_depth_error"]
        epoch_val_tp_mae = epoch_val_metrics["tp_depth_error"]

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epoch}: "
                f"Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}, "
                f"Train IoU: {epoch_train_iou:.4f}, Val IoU: {epoch_val_iou:.4f}, "
                f"Train MAE: {epoch_train_mae:.4f}, Val MAE: {epoch_val_mae:.4f}, "
                f"Train TP MAE: {epoch_train_tp_mae:.4f}, Val TP MAE: {epoch_val_tp_mae:.4f}")

    # Save the model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detection_model")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--depth_loss_weight", type=float, default=3.0)
    parser.add_argument("--dice_loss_weight", type=float, default=5.0)

    # Pass all arguments to train_detection
    train_detection(**vars(parser.parse_args()))
