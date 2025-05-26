import torch

def test_device():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA device properties
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        # Print information about each CUDA device
        for i in range(device_count):
            print(f"\nCUDA Device {i}:")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    else:
        print("Running on CPU")
    
    # Set device
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"\nUsing device: {device}")

if __name__ == "__main__":
    test_device()
