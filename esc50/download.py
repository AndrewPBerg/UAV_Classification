# Code to download the ESC-50 dataset from the official Github
# github: https://github.com/karolpiczak/ESC-50

import os
import urllib.request
import zipfile
import glob

def download_esc50():
    # URLs and paths
    github_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    download_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(download_dir, "ESC-50-master.zip")
    extract_dir = os.path.join(download_dir, "temp")
    dataset_dir = os.path.join(download_dir, "ESC-50")
    
    # Check if dataset already exists
    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, "meta", "esc50.csv")):
        print(f"ESC-50 dataset already exists at {dataset_dir}")
        return dataset_dir
    
    print("ESC-50 dataset not found or incomplete, downloading...")
    
    # Create temporary directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Downloading ESC-50 dataset from {github_url}...")
    
    # Download the zip file
    urllib.request.urlretrieve(github_url, zip_path)
    
    print(f"Download complete. Extracting files to {extract_dir}...")
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Source directory after extraction
    source_dir = os.path.join(extract_dir, "ESC-50-master")
    
    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Instead of using shutil, manually move files
    print(f"Moving files to {dataset_dir}...")
    
    # If destination exists, remove it first
    if os.path.exists(dataset_dir):
        # Remove all files and subdirectories in dataset_dir
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                for root, dirs, files in os.walk(item_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                os.rmdir(item_path)
    else:
        os.makedirs(dataset_dir)
    
    # Copy all files from source to destination
    for root, dirs, files in os.walk(source_dir):
        # Get the relative path from source_dir
        rel_path = os.path.relpath(root, source_dir)
        # Create the corresponding directory in dataset_dir
        if rel_path != '.':
            os.makedirs(os.path.join(dataset_dir, rel_path), exist_ok=True)
        
        # Copy all files in this directory
        for file in files:
            src_file = os.path.join(root, file)
            if rel_path == '.':
                dst_file = os.path.join(dataset_dir, file)
            else:
                dst_file = os.path.join(dataset_dir, rel_path, file)
            
            # Copy the file
            with open(src_file, 'rb') as f_src:
                with open(dst_file, 'wb') as f_dst:
                    f_dst.write(f_src.read())
    
    print(f"Dataset extracted to {dataset_dir}")
    
    # Clean up
    print("Cleaning up temporary files...")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    # Remove temp directory recursively
    if os.path.exists(extract_dir):
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(extract_dir)
    
    print("ESC-50 dataset download and extraction complete!")
    return dataset_dir

if __name__ == "__main__":
    download_esc50()

