import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from timeit import default_timer as timer
from tqdm.auto import tqdm
import wandb
import numpy as np
from typing import Dict, List

from .engine import train_step, test_step

def train_fold(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    epochs: int,
    device: str,
    num_classes: int,
    fold: int,
    accumulation_steps: int = 1,
    patience: int = 5,
    delta: float = 0.01,
    scaler: GradScaler | None = None
) -> Dict[str, List]:
    """Trains a single fold of k-fold cross validation"""
    
    if scaler is None:
        scaler = GradScaler()

    results = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": []
    }
    
    best_val_loss = float("inf")
    patience_counter = 0
    start = timer()

    # Initialize metrics for this fold
    precision_metric_train = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_train = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_train = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    precision_metric_val = MulticlassPrecision(num_classes=num_classes, average="weighted").to(device)
    recall_metric_val = MulticlassRecall(num_classes=num_classes, average="weighted").to(device)
    f1_metric_val = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)

    for epoch in tqdm(range(epochs), desc=f"Fold {fold+1} Training", leave=True):
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

        # Print metrics for this fold
        print(
            f"Fold {fold+1} Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Log metrics to wandb for this fold
        if wandb.run is not None:
            wandb.log({
                f"fold_{fold+1}/train_acc": train_acc,
                f"fold_{fold+1}/train_loss": train_loss,
                f"fold_{fold+1}/val_acc": val_acc,
                f"fold_{fold+1}/val_loss": val_loss,
                f"fold_{fold+1}/train_f1": train_f1,
                f"fold_{fold+1}/val_f1": val_f1,
                f"fold_{fold+1}/train_precision": train_precision,
                f"fold_{fold+1}/val_precision": val_precision,
                f"fold_{fold+1}/train_recall": train_recall,
                f"fold_{fold+1}/val_recall": val_recall,
                "epoch": epoch+1
            })

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["train_f1"].append(train_f1)
        results["val_f1"].append(val_f1)
        results["train_precision"].append(train_precision)
        results["val_precision"].append(val_precision)
        results["train_recall"].append(train_recall)
        results["val_recall"].append(val_recall)

        # Early stopping check
        if val_loss < best_val_loss - delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered in fold {fold+1} at epoch {epoch+1}")
            break

        # Step the scheduler
        scheduler.step(val_loss)

    end = timer()
    train_time = end - start
    hours = int(train_time // 3600)
    minutes = int((train_time % 3600) // 60)
    seconds = int(train_time % 60)
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    
    if wandb.run is not None:
        wandb.log({f"fold_{fold+1}/train_time": formatted_time})

    return results

def k_fold_cross_validation(
    model_fn,  # Function that creates a new model instance
    fold_datasets: list,
    optimizer_fn,  # Function that creates a new optimizer instance
    scheduler_fn,  # Function that creates a new scheduler instance
    loss_fn: nn.Module,
    device: str,
    num_classes: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    accumulation_steps: int = 1,
    patience: int = 5,
    scaler: GradScaler | None = None
) -> Dict[str, List]:
    """
    Performs k-fold cross validation training
    Returns aggregated results across all folds
    """
    all_fold_results = []
    
    for fold, (train_dataset, val_dataset) in enumerate(fold_datasets):
        print(f"\nTraining Fold {fold + 1}/{len(fold_datasets)}")
        
        # Create dataloaders for this fold
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )
        
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )
        
        # Create new model instance for this fold
        model = model_fn().to(device)
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer)
        
        # Train the fold
        fold_results = train_fold(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=epochs,
            device=device,
            num_classes=num_classes,
            fold=fold,
            accumulation_steps=accumulation_steps,
            patience=patience,
            scaler=scaler
        )
        
        all_fold_results.append(fold_results)
        
        # Log final metrics for this fold
        if wandb.run is not None:
            wandb.log({
                f"fold_{fold+1}_final_val_acc": fold_results["val_acc"][-1],
                f"fold_{fold+1}_final_val_loss": fold_results["val_loss"][-1],
                f"fold_{fold+1}_final_val_f1": fold_results["val_f1"][-1]
            })
    
    # Calculate and log average metrics across folds
    avg_metrics = calculate_average_metrics(all_fold_results)
    if wandb.run is not None:
        wandb.log(avg_metrics)
    
    return all_fold_results

def calculate_average_metrics(all_fold_results: List[Dict]) -> Dict:
    """Calculate average metrics across all folds"""
    avg_metrics = {}
    metrics = ["val_acc", "val_loss", "val_f1", "val_precision", "val_recall"]
    
    for metric in metrics:
        values = [fold[metric][-1] for fold in all_fold_results]  # Get final value for each fold
        avg_metrics[f"average_{metric}"] = np.mean(values)
        avg_metrics[f"std_{metric}"] = np.std(values)
    
    return avg_metrics 