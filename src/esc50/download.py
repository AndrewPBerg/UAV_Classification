import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Optional

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

def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def organize_esc50_dataset(base_path: str) -> None:
    """Organize ESC-50 dataset into class-based directory structure."""
    base_path_obj = Path(base_path)
    audio_dir = base_path_obj / "audio"
    meta_dir = base_path_obj / "meta"
    
    # Read the metadata CSV
    meta_csv = meta_dir / "esc50.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_csv}")
    
    df = pd.read_csv(meta_csv)
    
    # Create class directories
    classes_dir = base_path_obj / "classes"
    classes_dir.mkdir(exist_ok=True)
    
    print("Organizing files into class directories...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Organizing files"):
        category = row['category']
        filename = row['filename']
        
        # Create category directory if it doesn't exist
        category_dir = classes_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Source and destination paths
        src_file = audio_dir / filename
        dst_file = category_dir / filename
        
        # Copy file if it exists and destination doesn't exist
        if src_file.exists() and not dst_file.exists():
            import shutil
            shutil.copy2(src_file, dst_file)

def download_esc50() -> str:
    """
    Download and organize the ESC-50 dataset to src/datasets/esc50_dataset.
        
    Returns:
        Path to the organized dataset directory.
    """
    # Get the repository root (assumes this script is in src/esc50/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    data_dir = repo_root / "src" / "datasets"
    
    data_dir_obj = Path(data_dir)
    data_dir_obj.mkdir(parents=True, exist_ok=True)
    
    # ESC-50 dataset URL
    dataset_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_filename = "ESC-50-master.zip"
    zip_path = data_dir_obj / zip_filename
    
    print(f"Downloading ESC-50 dataset to {data_dir_obj}")
    
    # Download the dataset
    if not zip_path.exists():
        print("Downloading ESC-50 dataset...")
        download_file(dataset_url, str(zip_path))
    else:
        print("Dataset zip file already exists, skipping download.")
    
    # Extract the dataset
    extract_dir = data_dir_obj / "ESC-50-master"
    if not extract_dir.exists():
        print("Extracting dataset...")
        extract_zip(str(zip_path), str(data_dir_obj))
    else:
        print("Dataset already extracted, skipping extraction.")
    
    # Organize into class-based structure
    organize_esc50_dataset(str(extract_dir))
    
    # Clean up zip file
    if zip_path.exists():
        os.remove(zip_path)
        print("Cleaned up zip file.")
    
    organized_path = extract_dir / "classes"
    print(f"ESC-50 dataset downloaded and organized at: {organized_path}")
    
    return str(organized_path)

def main():
    """Main function to download ESC-50 dataset."""
    try:
        dataset_path = download_esc50()
        print(f"\nESC-50 dataset successfully downloaded and organized!")
        print(f"Dataset location: {dataset_path}")
        print(f"\nTo use this dataset, point your data_path to: {dataset_path}")
        
    except Exception as e:
        print(f"Error downloading ESC-50 dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 