# Distributed Data Parallel (DDP) Training Guide

This guide explains how to use the refactored codebase to train your UAV classification model with multiple GPUs using PyTorch Lightning's DDP strategy.

## Overview of Changes

1. Docker configuration:
   - Added shared memory size to support inter-process communication
   - Ensured all GPUs are visible to the container

2. PyTorch Lightning Trainer:
   - Automatically detects available GPUs and configures DDP strategy
   - Passes the correct accelerator, devices, and strategy parameters to trainer

3. Model code:
   - Removed explicit `.to(device)` and `.cuda()` calls to let PyTorch Lightning handle device placement
   - Updated to leverage the LightningModule's device attribute

## How to Run Multi-GPU Training

### 1. Start the Docker container:

```bash
docker-compose up --build
```

This will build and start the container with all GPUs available.

### 2. Verify GPU Availability:

When the code runs, it will print the number of available GPUs at the start:

```
Available GPUs: X, using strategy: ddp
```

Where X is the number of GPUs detected.

### 3. Monitor Training:

You can monitor training on multiple GPUs through:
- Terminal output that shows processes on different GPUs
- Weights & Biases logs if enabled
- NVIDIA-SMI command to check GPU utilization

## Best Practices for Multi-GPU Training

1. **Batch Size**: Consider increasing your batch size proportionally to the number of GPUs. 
   If you were using batch size 32 on a single GPU, you might use 32 * num_gpus for DDP.

2. **Learning Rate**: You might need to adjust the learning rate when scaling to multiple GPUs.
   A common approach is to use the "square root scaling rule": new_lr = old_lr * sqrt(num_gpus).

3. **Gradient Accumulation**: If you were using gradient accumulation with 1 GPU, you might 
   want to reduce the accumulation steps when using multiple GPUs.

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: 
   - Try reducing the batch size if you encounter OOM errors
   - Check that your data preprocessing is consistent across GPUs

2. **Process Communication Errors**:
   - Ensure the shared memory size is sufficient (set in docker-compose.yml)
   - Check for network connectivity issues between GPUs

3. **Performance Not Scaling**:
   - Ensure your data loading isn't a bottleneck (increase num_workers in DataLoader)
   - Consider using larger batch sizes to better utilize the GPUs

## Example Configuration

For optimal performance with 4 GPUs:

```python
# In your configuration file or parameters
batch_size = 32 * 4  # Scale with number of GPUs
learning_rate = base_lr * (4 ** 0.5)  # Apply square root scaling
num_workers = 4 * 4  # Scale workers with GPUs
```

## Further Optimization

Consider adding these optimizations for even better multi-GPU performance:

1. **Flash Attention**: If using transformer models, enable Flash Attention for faster training
2. **Mixed Precision**: Already enabled, uses less memory and speeds up training
3. **Gradient Checkpointing**: Consider enabling if model is large and hits memory limits 