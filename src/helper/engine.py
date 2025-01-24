import torch 
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy

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



def train_step(model, 
              dataloader, 
              loss_fn, 
              optimizer,
              device,
              scaler,
              precision_metric,
              recall_metric,
              f1_metric,
              accumulation_steps=1):
    model.train()
    train_loss = 0
    
    # Initialize accuracy metric
    accuracy_metric = MulticlassAccuracy(num_classes=precision_metric.num_classes, average="weighted").to(device)
    
    len_dataloader = len(dataloader)
    optimizer.zero_grad()
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Ensure input is float32 before forward pass
        X = X.float()
        
        # Forward pass with autocast
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # print(f"X shape: {X.shape}")
            # print(f"X: {X}")
            # print(f"X.shape: {X.shape}")
            outputs = model(X)
            if hasattr(outputs, "logits"):
                y_pred = outputs.logits
            else:
                y_pred = outputs
            loss = loss_fn(y_pred, y)
            loss = loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len_dataloader):
            # Unscale gradients and clip them
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        train_loss += loss.item() * accumulation_steps
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        # Update metrics
        accuracy_metric.update(y_pred_class, y)
        precision_metric.update(y_pred_class, y)
        recall_metric.update(y_pred_class, y)
        f1_metric.update(y_pred_class, y)

    train_loss /= len(dataloader)
    train_acc = accuracy_metric.compute().item()
    train_precision = precision_metric.compute().item()
    train_recall = recall_metric.compute().item()
    train_f1 = f1_metric.compute().item()

    # Reset metrics
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    return train_loss, train_acc, train_precision, train_recall, train_f1


def test_step(model, dataloader, loss_fn, device, precision_metric, recall_metric, f1_metric):
    model.eval()
    test_loss = 0

    # Initialize accuracy metric
    accuracy_metric = MulticlassAccuracy(num_classes=precision_metric.num_classes, average="weighted").to(device)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.float()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(X)
                if hasattr(outputs, "logits"):
                    y_pred = outputs.logits
                else:
                    y_pred = outputs
                loss = loss_fn(y_pred, y)

            test_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

            # Update metrics
            accuracy_metric.update(y_pred_class, y)
            precision_metric.update(y_pred_class, y)
            recall_metric.update(y_pred_class, y)
            f1_metric.update(y_pred_class, y)

    test_loss /= len(dataloader)
    test_acc = accuracy_metric.compute().item()
    test_precision = precision_metric.compute().item()
    test_recall = recall_metric.compute().item()
    test_f1 = f1_metric.compute().item()

    # Reset metrics
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    return test_loss, test_acc, test_precision, test_recall, test_f1


def train(model: torch.nn.Module, 
          train_dataloader: DataLoader, 
          val_dataloader: DataLoader,
          test_dataloader: DataLoader | None, 
          optimizer: AdamW, 
          scheduler: torch.optim.lr_scheduler._LRScheduler,  
          loss_fn: nn.Module, 
          epochs: int, 
          device: str, 
          num_classes: int,
          accumulation_steps: int = 1,  
          patience: int = 5, 
          delta: float = 0.01,
          scaler: GradScaler | None = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model with additional metrics (F1, precision, recall) using torchmetrics."""
    
    if scaler is None:
        scaler = GradScaler()

    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": [], 
               "train_f1": [], "val_f1": [], "test_f1": [], 
               "train_precision": [], "val_precision": [], "test_precision": [], 
               "train_recall": [], "val_recall": [], "test_recall": []}
    best_loss = float("inf")  
    patience_counter = 0  
    start = timer()

    # Initialize torchmetrics metrics for training
    precision_metric_train = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_train = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_train = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    # Initialize torchmetrics metrics for training validation
    precision_metric_val = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_val = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_val = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    # Initialize torchmetrics metrics for testing
    precision_metric_test = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_test = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_test = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    for epoch in tqdm(range(epochs), dynamic_ncols=True):
        # Training step
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(
            model=model, 
            dataloader=train_dataloader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device, 
            accumulation_steps=accumulation_steps,
            precision_metric=precision_metric_train,
            recall_metric=recall_metric_train,
            f1_metric=f1_metric_train,
            scaler=scaler
        )
        
        # Validation step
        val_loss, val_acc, val_precision, val_recall, val_f1 = test_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
            precision_metric=precision_metric_val,
            recall_metric=recall_metric_val,
            f1_metric=f1_metric_val
        )
        
        # Test step (only if test_dataloader is provided)
        if test_dataloader is not None:
            test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
                precision_metric=precision_metric_test,
                recall_metric=recall_metric_test,
                f1_metric=f1_metric_test
            )
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            results["test_f1"].append(test_f1)
            results["test_precision"].append(test_precision)
            results["test_recall"].append(test_recall)

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
                "val_acc": val_acc,  # Add validation accuracy
                "val_loss": val_loss,  # Add validation loss
                "test_acc": test_acc, 
                "test_loss": test_loss, 
                "train_f1": train_f1,
                "val_f1": val_f1,  # Add validation F1 score
                "test_f1": test_f1,
                "train_precision": train_precision,
                "val_precision": val_precision,  # Add validation precision
                "test_precision": test_precision,
                "train_recall": train_recall,
                "val_recall": val_recall,  # Add validation recall
                "test_recall": test_recall,
                "epoch": epoch+1
            })

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_f1"].append(train_f1)
        results["val_f1"].append(val_f1)
        results["test_f1"].append(test_f1)
        results["train_precision"].append(train_precision)
        results["val_precision"].append(val_precision)
        results["test_precision"].append(test_precision)
        results["train_recall"].append(train_recall)
        results["val_recall"].append(val_recall)
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
    total_train_time = end - start
    # Convert the elapsed time to hrs:min:sec format
    hours = int(total_train_time // 3600)
    minutes = int((total_train_time % 3600) // 60)
    seconds = int(total_train_time % 60)
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"Train time on {device}: {total_train_time}")
    if wandb.run is not None:
        wandb.log({"train_time": formatted_time})


        
    return results

def inference_loop(model: torch.nn.Module, 
                  inference_loader: DataLoader,
                  loss_fn: nn.CrossEntropyLoss,
                  device: str,
                  num_classes: int) -> None:
    """
    Performs inference on a trained model using mixed precision.
    Uses torchmetrics for accurate metric calculation.
    """
    
    # Initialize metrics
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="weighted").to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)
    
    total_loss = 0.0
    
    # Set model to evaluation mode
    model.eval()
    
    # Mixed precision and no gradients during inference
    with torch.no_grad():
        for X_batch, y_batch in inference_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Enable autocast for mixed precision
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(X_batch)
                # Get logits from the model
                if hasattr(outputs, "logits"):
                    y_pred = outputs.logits
                else:
                    y_pred = outputs

                # Calculate loss
                loss = loss_fn(y_pred, y_batch)
                total_loss += loss.item()

                # Get predictions
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                
                # Update metrics
                accuracy_metric.update(y_pred_class, y_batch)
                precision_metric.update(y_pred_class, y_batch)
                recall_metric.update(y_pred_class, y_batch)
                f1_metric.update(y_pred_class, y_batch)

    # Calculate final metrics
    average_loss = total_loss / len(inference_loader)
    accuracy = accuracy_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()

    # Reset metrics
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    print(
        f"Inference Results:\n"
        f"Loss: {average_loss:.4f}\n"
        f"Accuracy: {accuracy * 100:.2f}%\n"
        f"Precision: {precision * 100:.2f}%\n"
        f"Recall: {recall * 100:.2f}%\n"
        f"F1 Score: {f1 * 100:.2f}%"
    )

    if wandb.run is not None:
        wandb.log({
            "inference_loss": average_loss,
            "inference_accuracy": accuracy,
            "inference_precision": precision,
            "inference_recall": recall,
            "inference_f1": f1
        })