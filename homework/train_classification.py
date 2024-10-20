import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import Classifier, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric


def train(
        exp_dir: str = 'logs',
        model_name: str = 'cnn_classifier',
        num_epoch: int = 100,
        lr: float = 1e-3, 
        batch_size: int = 128,
        seed: int = 2024,
        **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a log directory with a timestamp
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    #initialize the model and send it to the device
    model = Classifier(**kwargs)
    model = model.to(device)

    #load training and validation data with data augmentation for the training set
    train_data = load_data("classification_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    #create loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #define metric for accuracy tracking
    accuracy_metric = AccuracyMetric()

    global_step = 0

    metrics = {'train_acc':[], 'val_acc':[], 'train_loss':[]}

    #Trainig loop
    for epoch in range(num_epoch):
        #clear metrics at the beginning of every epoch
        for key in metrics:
            metrics[key].clear()

        accuracy_metric.reset()  # Reset metrics at the start of each epoch

        model.train()

        for img,label in train_data:
            img, label = img.to(device), label.to(device)

            #Training Step
            preds = model(img)
            loss = loss_func(preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update accuracy metric with training predictions
            _, predicted = torch.max(preds, 1)
            accuracy_metric.add(predicted, label)

            global_step += 1

        # Compute epoch training accuracy
        epoch_train_acc = accuracy_metric.compute()["accuracy"]

        #Validation loop
        with torch.no_grad():
            model.eval()
            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                val_preds = model(img)
                _, val_predicted = torch.max(val_preds, 1)
                accuracy_metric.add(val_predicted, label)


        # Compute epoch validation accuracy
        epoch_val_acc = accuracy_metric.compute()["accuracy"]

        # Print metrics
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epoch}: train_acc={epoch_train_acc:.4f}, val_acc={epoch_val_acc:.4f}")

    # Save the model
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="cnn_classifier")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)

    # Pass all arguments to train
    train(**vars(parser.parse_args()))


