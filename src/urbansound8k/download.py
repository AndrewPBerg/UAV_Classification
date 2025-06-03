import os
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Optional
import shutil

def download_file(url: str, destination: str) -> None:
    """Download a file from URL with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

def extract_tar(tar_path: str, extract_to: str) -> None:
    """Extract a tar.gz file to the specified directory."""
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)

def organize_urbansound8k_dataset(base_path: str) -> None:
    """Organize UrbanSound8K dataset into class-based directory structure."""
    base_path_obj = Path(base_path)
    audio_dir = base_path_obj / "audio"
    metadata_dir = base_path_obj / "metadata"
    
    # Read the metadata CSV
    meta_csv = metadata_dir / "UrbanSound8K.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_csv}")
    
    df = pd.read_csv(meta_csv)
    
    # Create class directories
    classes_dir = base_path_obj / "classes"
    classes_dir.mkdir(exist_ok=True)
    
    print("Organizing files into class directories...")
    
    # Get unique class names
    unique_classes = df['class'].unique()
    print(f"Found {len(unique_classes)} classes: {sorted(unique_classes)}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing files"):
        class_name = row['class']
        fold = row['fold']
        filename = row['slice_file_name']
        
        # Create class directory if it doesn't exist
        class_dir = classes_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Source path (UrbanSound8K stores files in fold directories)
        src_file = audio_dir / f"fold{fold}" / filename
        dst_file = class_dir / filename
        
        # Copy file if it exists and destination doesn't exist
        if src_file.exists() and not dst_file.exists():
            shutil.copy2(src_file, dst_file)
        elif not src_file.exists():
            print(f"Warning: Source file not found: {src_file}")

def download_urbansound8k() -> str:
    """
    Download and organize the UrbanSound8K dataset to src/datasets/urbansound8k_dataset.
        
    Returns:
        Path to the organized dataset directory.
    """
    # Get the repository root (assumes this script is in src/urbansound8k/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    data_dir = repo_root / "src" / "datasets"
    
    data_dir_obj = Path(data_dir)
    data_dir_obj.mkdir(parents=True, exist_ok=True)
    
    # UrbanSound8K dataset URL
    dataset_url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
    tar_filename = "UrbanSound8K.tar.gz"
    tar_path = data_dir_obj / tar_filename
    
    print(f"Downloading UrbanSound8K dataset to {data_dir_obj}")
    
    # Download the dataset
    if not tar_path.exists():
        print("Downloading UrbanSound8K dataset...")
        download_file(dataset_url, str(tar_path))
    else:
        print("Dataset tar file already exists, skipping download.")
    
    # Extract the dataset
    extract_dir = data_dir_obj / "UrbanSound8K"
    if not extract_dir.exists():
        print("Extracting dataset...")
        extract_tar(str(tar_path), str(data_dir_obj))
    else:
        print("Dataset already extracted, skipping extraction.")
    
    # Organize into class-based structure
    organize_urbansound8k_dataset(str(extract_dir))
    
    # Clean up tar file
    if tar_path.exists():
        os.remove(tar_path)
        print("Cleaned up tar file.")
    
    organized_path = extract_dir / "classes"
    print(f"UrbanSound8K dataset downloaded and organized at: {organized_path}")
    
    return str(organized_path)

def main():
    """Main function to download UrbanSound8K dataset."""
    try:
        dataset_path = download_urbansound8k()
        print(f"\nUrbanSound8K dataset successfully downloaded and organized!")
        print(f"Dataset location: {dataset_path}")
        print(f"\nTo use this dataset, point your data_path to: {dataset_path}")
        
        # Print dataset statistics
        df = pd.read_csv(Path(dataset_path).parent / "metadata" / "UrbanSound8K.csv")
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Number of classes: {df['class'].nunique()}")
        print(f"Number of folds: {df['fold'].nunique()}")
        print(f"Classes: {sorted(df['class'].unique())}")
        
    except Exception as e:
        print(f"Error downloading UrbanSound8K dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 