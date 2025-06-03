import os
import zipfile
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import shutil
from typing import Optional

def check_kagglehub_setup() -> bool:
    """Check if kagglehub is properly set up."""
    try:
        import kagglehub
        return True
    except ImportError:
        print("kagglehub package not found. Install with: pip install kagglehub")
        return False

def download_audiomnist_kagglehub(destination_dir: str) -> str:
    """
    Download AudioMNIST dataset using kagglehub.
    
    Args:
        destination_dir: Directory to download and extract the dataset
        
    Returns:
        Path to the extracted dataset directory
    """
    if not check_kagglehub_setup():
        raise RuntimeError("kagglehub not properly set up")
    
    import kagglehub
    
    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)
    
    # AudioMNIST dataset identifier - using the correct dataset reference
    dataset_name = "sripaadsrinivasan/audio-mnist"
    
    print(f"Downloading AudioMNIST dataset using kagglehub: {dataset_name}")
    print(f"Destination: {destination_path}")
    
    try:
        # Download the dataset using kagglehub
        downloaded_path = kagglehub.dataset_download(dataset_name)
        
        print(f"Dataset downloaded to: {downloaded_path}")
        
        # Copy the downloaded files to our desired location
        downloaded_path_obj = Path(downloaded_path)
        
        if downloaded_path_obj.exists():
            # Copy all files from the downloaded location to our destination
            if downloaded_path_obj.is_file():
                # Single file download
                shutil.copy2(downloaded_path_obj, destination_path)
                return str(destination_path)
            else:
                # Directory download - copy contents
                for item in downloaded_path_obj.rglob("*"):
                    if item.is_file():
                        relative_path = item.relative_to(downloaded_path_obj)
                        target_path = destination_path / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target_path)
                
                print("AudioMNIST dataset downloaded and copied successfully!")
                return str(destination_path)
        else:
            raise FileNotFoundError(f"Downloaded path does not exist: {downloaded_path}")
            
    except Exception as e:
        print(f"Error downloading AudioMNIST dataset: {e}")
        raise

def check_kaggle_setup() -> bool:
    """Check if Kaggle API is properly set up."""
    try:
        import kaggle
        return True
    except ImportError:
        print("Kaggle package not found. Install with: pip install kaggle")
        return False
    except OSError as e:
        if "Could not find kaggle.json" in str(e):
            print("Kaggle API credentials not found.")
            print("Please set up your Kaggle API credentials:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_CONFIG_DIR")
            return False
        raise

def setup_project_kaggle_config():
    """Set up Kaggle configuration to use project-local kaggle.json file."""
    # Get the path to the project's .kaggle directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    kaggle_config_dir = repo_root / "src" / ".kaggle"
    
    if kaggle_config_dir.exists() and (kaggle_config_dir / "kaggle.json").exists():
        # Set environment variable to point to project-local config
        os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_config_dir)
        print(f"Using project-local Kaggle config: {kaggle_config_dir / 'kaggle.json'}")
        return True
    else:
        print(f"Project-local kaggle.json not found at: {kaggle_config_dir / 'kaggle.json'}")
        return False

def download_audiomnist_kaggle(destination_dir: str) -> str:
    """
    Download AudioMNIST dataset from Kaggle.
    
    Args:
        destination_dir: Directory to download and extract the dataset
        
    Returns:
        Path to the extracted dataset directory
    """
    # First try to set up project-local Kaggle config
    if not setup_project_kaggle_config():
        print("Falling back to system-wide Kaggle configuration...")
    
    if not check_kaggle_setup():
        raise RuntimeError("Kaggle API not properly set up")
    
    import kaggle
    
    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)
    
    # AudioMNIST dataset identifier on Kaggle
    dataset_name = "soerenab/audiomnist"
    
    print(f"Downloading AudioMNIST dataset from Kaggle: {dataset_name}")
    print(f"Destination: {destination_path}")
    
    try:
        # Download the dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(destination_path),
            unzip=True
        )
        
        print("AudioMNIST dataset downloaded and extracted successfully!")
        
        # Find the extracted directory
        extracted_dirs = [d for d in destination_path.iterdir() if d.is_dir()]
        if extracted_dirs:
            return str(extracted_dirs[0])
        else:
            return str(destination_path)
            
    except Exception as e:
        print(f"Error downloading AudioMNIST dataset: {e}")
        raise

def organize_audiomnist_dataset(base_path: str) -> None:
    """
    Organize AudioMNIST dataset into class-based directory structure.
    Files are organized by digit class (0-9) based on the first character of the filename.
    Example: "2_01_12.wav" goes into the "2" directory.
    """
    base_path_obj = Path(base_path)
    
    print("Organizing AudioMNIST dataset by digit class...")
    
    # Check if we already have a proper class-based structure
    class_dirs = [d for d in base_path_obj.iterdir() if d.is_dir() and d.name.isdigit()]
    
    # Count files in existing class directories
    existing_class_files = 0
    for class_dir in class_dirs:
        existing_class_files += len(list(class_dir.glob("*.wav")))
    
    # Look for .wav files in the base directory (need to be organized)
    audio_files = list(base_path_obj.glob("*.wav"))
    
    if len(class_dirs) == 10 and existing_class_files > 0 and len(audio_files) == 0:
        print("AudioMNIST dataset already properly organized with 10 digit classes.")
        return
    
    if not audio_files:
        # Check if files might be in subdirectories that need to be flattened
        all_audio_files = list(base_path_obj.rglob("*.wav"))
        if all_audio_files:
            print(f"Found {len(all_audio_files)} audio files in subdirectories. Flattening structure...")
            # Move all files to base directory first
            for audio_file in all_audio_files:
                if audio_file.parent != base_path_obj:
                    target_file = base_path_obj / audio_file.name
                    if not target_file.exists():
                        shutil.move(str(audio_file), str(target_file))
            # Update the audio_files list
            audio_files = list(base_path_obj.glob("*.wav"))
        else:
            print("No .wav files found. Dataset may already be organized or missing.")
            return
    
    print(f"Found {len(audio_files)} audio files to organize into digit classes...")
    
    # Create digit directories (0-9)
    for digit in range(10):
        digit_dir = base_path_obj / str(digit)
        digit_dir.mkdir(exist_ok=True)
    
    # Organize files by digit class based on first character of filename
    moved_count = 0
    error_count = 0
    
    for audio_file in tqdm(audio_files, desc="Organizing files by digit class"):
        filename = audio_file.name
        
        try:
            # Extract digit from filename (first character should be the digit)
            digit_char = filename[0]
            digit = int(digit_char)
            
            if 0 <= digit <= 9:
                target_dir = base_path_obj / str(digit)
                target_file = target_dir / filename
                
                if not target_file.exists():
                    shutil.move(str(audio_file), str(target_file))
                    moved_count += 1
                else:
                    print(f"Warning: File already exists, skipping: {target_file}")
            else:
                print(f"Warning: Invalid digit {digit} in filename: {filename}")
                error_count += 1
                
        except (ValueError, IndexError):
            print(f"Warning: Could not extract digit from filename: {filename}")
            error_count += 1
    
    print(f"Organization complete:")
    print(f"  ✅ Moved {moved_count} files into digit-based directories")
    if error_count > 0:
        print(f"  ⚠️  {error_count} files could not be organized")
    
    # Verify organization
    print("\nVerifying organization:")
    total_organized = 0
    for digit in range(10):
        digit_dir = base_path_obj / str(digit)
        if digit_dir.exists():
            file_count = len(list(digit_dir.glob("*.wav")))
            total_organized += file_count
            print(f"  Class {digit}: {file_count} files")
    
    print(f"  Total organized files: {total_organized}")
    
    # Clean up any empty subdirectories
    for item in base_path_obj.iterdir():
        if item.is_dir() and not item.name.isdigit():
            try:
                if not any(item.iterdir()):  # Empty directory
                    item.rmdir()
                    print(f"Removed empty directory: {item.name}")
            except:
                pass  # Directory not empty or other error

def download_audiomnist() -> str:
    """
    Download and organize the AudioMNIST dataset to src/datasets/audiomnist_dataset.
    
    Returns:
        Path to the organized dataset directory.
    """
    # Get the repository root (assumes this script is in src/audiomnist/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    data_dir = repo_root / "src" / "datasets"
    
    data_dir_obj = Path(data_dir)
    data_dir_obj.mkdir(parents=True, exist_ok=True)
    
    audiomnist_dir = data_dir_obj / "audiomnist_dataset"
    
    print(f"Downloading AudioMNIST dataset to {audiomnist_dir}")
    
    # Download the dataset
    if not audiomnist_dir.exists() or not any(audiomnist_dir.iterdir()):
        print("Downloading AudioMNIST dataset using kagglehub...")
        try:
            # Try kagglehub first (newer, more reliable method)
            extracted_path = download_audiomnist_kagglehub(str(audiomnist_dir))
                
        except Exception as e:
            print(f"Error downloading with kagglehub: {e}")
            print("Falling back to regular Kaggle API...")
            try:
                extracted_path = download_audiomnist_kaggle(str(audiomnist_dir))
                
                # If the dataset was extracted to a subdirectory, move contents up
                extracted_path_obj = Path(extracted_path)
                if extracted_path_obj != audiomnist_dir and extracted_path_obj.parent == audiomnist_dir:
                    # Move contents from subdirectory to main directory
                    for item in extracted_path_obj.iterdir():
                        shutil.move(str(item), str(audiomnist_dir / item.name))
                    extracted_path_obj.rmdir()
                    
            except Exception as e2:
                print(f"Error downloading from Kaggle API: {e2}")
                print("Please download AudioMNIST manually and place it in the dataset directory.")
                raise
    else:
        print("AudioMNIST dataset directory already exists, skipping download.")
    
    # Organize into class-based structure
    organize_audiomnist_dataset(str(audiomnist_dir))
    
    print(f"AudioMNIST dataset downloaded and organized at: {audiomnist_dir}")
    
    return str(audiomnist_dir)

def main():
    """Main function to download AudioMNIST dataset."""
    try:
        dataset_path = download_audiomnist()
        print(f"\nAudioMNIST dataset successfully downloaded and organized!")
        print(f"Dataset location: {dataset_path}")
        print(f"\nTo use this dataset, point your data_path to: {dataset_path}")
        
        # Quick verification
        dataset_path_obj = Path(dataset_path)
        class_dirs = [d for d in dataset_path_obj.iterdir() if d.is_dir() and d.name.isdigit()]
        total_files = sum(len(list(d.glob("*.wav"))) for d in class_dirs)
        
        print(f"\nDataset verification:")
        print(f"  Found {len(class_dirs)} digit classes")
        print(f"  Total audio files: {total_files}")
        
        if len(class_dirs) == 10:
            print("✅ Dataset appears to be properly organized!")
        else:
            print("⚠️ Warning: Expected 10 digit classes, found {len(class_dirs)}")
        
    except Exception as e:
        print(f"Error downloading AudioMNIST dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 