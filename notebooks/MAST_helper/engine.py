import torch 
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import wandb
from torch.amp import autocast, GradScaler
import torch.optim

# Set up mixed precision scaler
scaler = GradScaler()

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


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              device: str) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch."""

    model.eval()  # Set model to evaluation mode
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass (mixed precision not needed during evaluation)
            outputs = model(X)
            test_pred_logits = outputs.logits  # Get the logits

            # Calculate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # Average loss and accuracy
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler._LRScheduler,  # Add scheduler here
          loss_fn: nn.Module, 
          epochs: int, 
          device: str, 
          accumulation_steps: int = 1,  # Gradient accumulation steps
          patience: int = 5, 
          delta: float = 0.01) -> Dict[str, List]:
    """Trains and tests a PyTorch model with gradient accumulation, early stopping, and learning rate scheduling."""
    
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_loss = float("inf")  # Initialize best loss to a high value
    patience_counter = 0  # Counter for early stopping

    for epoch in tqdm(range(epochs)):
        # Set model to training mode
        model.train()
        train_loss, train_acc = 0, 0
        optimizer.zero_grad()

        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # Forward pass
            outputs = model(X)
            y_pred = outputs.logits

            # Calculate loss
            loss = loss_fn(y_pred, y)
            loss = loss / accumulation_steps  # Scale loss by accumulation steps
            loss.backward()

            # Perform optimizer step after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Track train loss and accuracy
            train_loss += loss.item() * accumulation_steps  # Multiply back for logging
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Calculate train metrics
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # Test the model
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Print metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

        if wandb.run is not None:
            # Log metrics to wandb
            wandb.log({
                "train_acc": train_acc, 
                "train_loss": train_loss, 
                "test_acc": test_acc, 
                "test_loss": test_loss, 
                "epoch": epoch+1
            })

        # Store results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Early stopping logic
        if test_loss < best_loss - delta:  # Improvement criterion
            best_loss = test_loss  # Update best loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.")
            break

        # Step the scheduler (based on validation loss for ReduceLROnPlateau)
        scheduler.step(test_loss)  # Adjust learning rate based on validation loss

    return results

def inference_loop(model: torch.nn.Module, 
                   inference_loader: DataLoader,
                   loss_fn: nn.CrossEntropyLoss,
                   device: str):
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
