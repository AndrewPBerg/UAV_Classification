import torch 
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import wandb
import wandb.plot as plot
from torch.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
import torch.optim
# from torch.optim.adamw import AdamW # type: ignore
from torch.optim import AdamW # type: ignore
from timeit import default_timer as timer 


# Set up mixed precision scaler
scaler = GradScaler()
ground_truth_all, predictions_all = [], []

def wandb_plot_confusion_matrix(y_truth, preds, class_names) -> None:
    cm = plot.confusion_matrix(
        y_true=y_truth, preds=preds, class_names=class_names
    )
    
    wandb.log({"confusion matrix": cm})



def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, # type: ignore
               device: str,
               accumulation_steps: int) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch with mixed precision and gradient accumulation."""

    model.train()  # Set model to training mode
    train_loss, train_acc = 0, 0
    
    # Loop through batches
    optimizer.zero_grad()  # Zero the gradients initially
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(X)
            y_pred = outputs.logits  # Get the logits from SequenceClassifierOutput

            # Calculate loss and scale by accumulation steps
            loss = loss_fn(y_pred, y) / accumulation_steps
            train_loss += loss.item()

        # Backward pass with scaling
        scaler.scale(loss).backward()

        # Update the model weights every `accumulation_steps` batches
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Average loss and accuracy
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device, precision_metric, recall_metric, f1_metric):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            y_pred = outputs.logits

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)

            # Update torchmetrics for test set
            precision_metric.update(y_pred_class, y)
            recall_metric.update(y_pred_class, y)
            f1_metric.update(y_pred_class, y)

            ground_truth_all.extend(y.cpu().numpy())
            predictions_all.extend(y_pred_class.cpu().numpy())

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    # Compute metrics
    test_precision = precision_metric.compute().item()
    test_recall = recall_metric.compute().item()
    test_f1 = f1_metric.compute().item()

    # Reset metrics for the next epoch
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    return test_loss, test_acc, test_precision, test_recall, test_f1, 


def train(model: torch.nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          optimizer: AdamW, 
          scheduler: torch.optim.lr_scheduler._LRScheduler,  
          loss_fn: nn.Module, 
          epochs: int, 
          device: str, 
          num_classes: int,
          accumulation_steps: int = 1,  
          patience: int = 5, 
          delta: float = 0.01) -> Dict[str, List]:
    """Trains and tests a PyTorch model with additional metrics (F1, precision, recall) using torchmetrics."""
    
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], 
               "train_f1": [], "test_f1": [], "train_precision": [], "test_precision": [], 
               "train_recall": [], "test_recall": []}
    best_loss = float("inf")  
    patience_counter = 0  
    start = timer()

    # Initialize torchmetrics metrics for training
    precision_metric_train = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_train = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_train = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    # Initialize torchmetrics metrics for testing
    precision_metric_test = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_test = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_test = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    for epoch in tqdm(range(epochs), dynamic_ncols=True):
        model.train()
        train_loss, train_acc = 0, 0
        optimizer.zero_grad()

        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            y_pred = outputs.logits

            loss = loss_fn(y_pred, y) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps  
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

            # Update torchmetrics for train set
            precision_metric_train.update(y_pred_class, y)
            recall_metric_train.update(y_pred_class, y)
            f1_metric_train.update(y_pred_class, y)

        # Calculate train metrics
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        train_precision = precision_metric_train.compute().item()
        train_recall = recall_metric_train.compute().item()
        train_f1 = f1_metric_train.compute().item()

        # Reset metrics for the next epoch
        precision_metric_train.reset()
        recall_metric_train.reset()
        f1_metric_train.reset()

        # Test the model
        test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            precision_metric=precision_metric_test,
            recall_metric=recall_metric_test,
            f1_metric=f1_metric_test
        )

        # Print metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Test F1: {test_f1:.4f}"
        )

        if wandb.run is not None:
            # Log metrics to wandb
            wandb.log({
                "train_acc": train_acc, 
                "train_loss": train_loss, 
                "test_acc": test_acc, 
                "test_loss": test_loss, 
                "train_f1": train_f1,
                "test_f1": test_f1,
                "train_precision": train_precision,
                "test_precision": test_precision,
                "train_recall": train_recall,
                "test_recall": test_recall,
                "epoch": epoch+1
            })

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_f1"].append(train_f1)
        results["test_f1"].append(test_f1)
        results["train_precision"].append(train_precision)
        results["test_precision"].append(test_precision)
        results["train_recall"].append(train_recall)
        results["test_recall"].append(test_recall)

        # Early stopping logic
        if test_loss < best_loss - delta:  
            best_loss = test_loss  
            patience_counter = 0  
        else:
            patience_counter += 1  

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.")
            break

        # Step the scheduler (based on validation loss for ReduceLROnPlateau)
        scheduler.step(test_loss)   # type: ignore

    end = timer()
    total_time = end - start
    # Convert the elapsed time to hrs:min:sec format
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"Train time on {device}: {total_time}")
    if wandb.run is not None:
        wandb.log({"train_time": formatted_time})

        wandb_plot_confusion_matrix(y_truth=ground_truth_all,
                              preds=predictions_all,
                              class_names=train_dataloader.dataset.get_classes()) # type: ignore

        
    return results

def inference_loop(model: torch.nn.Module, 
                   inference_loader: DataLoader,
                   loss_fn: nn.CrossEntropyLoss,
                   device: str) -> None:
    """Performs inference on a trained model using mixed precision."""
    
    total_loss, correct, total = 0.0, 0, 0

    # Set model to evaluation mode
    model.eval()
    
    # Mixed precision and no gradients during inference
    with torch.no_grad():
        for X_batch, y_batch in inference_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Enable autocast for mixed precision
            with torch.cuda.amp.autocast():
                # Forward pass (extract logits)
                outputs = model(X_batch)
                logits = outputs.logits  # Get logits from the model

                # Calculate loss
                loss = loss_fn(logits, y_batch)
                total_loss += loss.item()

                # Get predictions
                _, predicted = torch.max(logits, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

    # Average loss and accuracy
    average_loss = total_loss / len(inference_loader)
    accuracy = correct / total

    print(f"Inference Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    if wandb.run is not None:
        wandb.log({
            "inference_loss": average_loss,
            "inference_accuracy": accuracy
        })



def sweep_train(model: torch.nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          optimizer: AdamW, 
          scheduler: torch.optim.lr_scheduler._LRScheduler,  
          loss_fn: nn.Module, 
          epochs: int, 
          device: str, 
          num_classes: int,
          accumulation_steps: int = 1,  
          patience: int = 5, 
          delta: float = 0.01) -> Dict[str, List]:

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], 
               "train_f1": [], "test_f1": [], "train_precision": [], "test_precision": [], 
               "train_recall": [], "test_recall": []}
    start = timer()

    # Initialize torchmetrics metrics for training
    precision_metric_train = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_train = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_train = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    # Initialize torchmetrics metrics for testing
    precision_metric_test = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_test = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_test = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss, train_acc = 0, 0
        optimizer.zero_grad()

        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            y_pred = outputs.logits

            loss = loss_fn(y_pred, y) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps  
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

            # Update torchmetrics for train set
            precision_metric_train.update(y_pred_class, y)
            recall_metric_train.update(y_pred_class, y)
            f1_metric_train.update(y_pred_class, y)

        # Calculate train metrics
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        train_precision = precision_metric_train.compute().item()
        train_recall = recall_metric_train.compute().item()
        train_f1 = f1_metric_train.compute().item()

        # Reset metrics for the next epoch
        precision_metric_train.reset()
        recall_metric_train.reset()
        f1_metric_train.reset()

        # Test the model
        test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            precision_metric=precision_metric_test,
            recall_metric=recall_metric_test,
            f1_metric=f1_metric_test
        )

        # Print metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Test F1: {test_f1:.4f}"
        )

        if wandb.run is not None:
            # Log metrics to wandb
            wandb.log({
                "train_acc": train_acc, 
                "train_loss": train_loss, 
                "test_acc": test_acc, 
                "test_loss": test_loss, 
                "train_f1": train_f1,
                "test_f1": test_f1,
                "train_precision": train_precision,
                "test_precision": test_precision,
                "train_recall": train_recall,
                "test_recall": test_recall,
                "epoch": epoch+1
            })

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_f1"].append(train_f1)
        results["test_f1"].append(test_f1)
        results["train_precision"].append(train_precision)
        results["test_precision"].append(test_precision)
        results["train_recall"].append(train_recall)
        results["test_recall"].append(test_recall)
        
        # Step the scheduler (based on validation loss for ReduceLROnPlateau)
        scheduler.step(test_loss) # type: ignore

    end = timer()
    total_time = end - start
    # Convert the elapsed time to hrs:min:sec format
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"Train time on {device}: {total_time}")
    if wandb.run is not None:
        wandb.log({"train_time": formatted_time})
    return results

    # results = train(model,
    #                 train_dataloader=train_loader,
    #                 test_dataloader=test_loader,
    #                 optimizer=optimizer,
    #                 # scheduler=scheduler,
    #                 loss_fn=criterion,
    #                 epochs=EPOCHS,
    #                 device=device,

# def wb_confmat(gorund_truth, )