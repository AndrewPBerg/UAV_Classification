import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, List

def check_esc50_availability(base_path: Optional[str] = None) -> Dict[str, any]:
    """
    Check if ESC-50 dataset is available and properly structured.
    
    Args:
        base_path: Optional custom path to check. If None, uses default location.
        
    Returns:
        Dictionary with availability status and paths
    """
    if base_path is None:
        # Get the repository root (assumes this script is in src/esc10/)
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        base_path = repo_root / "src" / "datasets" / "ESC-50-master"
    
    base_path_obj = Path(base_path)
    
    result = {
        "available": False,
        "base_path": str(base_path_obj),
        "audio_dir": None,
        "meta_file": None,
        "classes_dir": None,
        "issues": []
    }
    
    # Check if base directory exists
    if not base_path_obj.exists():
        result["issues"].append(f"ESC-50 base directory not found: {base_path_obj}")
        return result
    
    # Check for audio directory
    audio_dir = base_path_obj / "audio"
    if not audio_dir.exists():
        result["issues"].append(f"ESC-50 audio directory not found: {audio_dir}")
    else:
        result["audio_dir"] = str(audio_dir)
    
    # Check for metadata file
    meta_file = base_path_obj / "meta" / "esc50.csv"
    if not meta_file.exists():
        result["issues"].append(f"ESC-50 metadata file not found: {meta_file}")
    else:
        result["meta_file"] = str(meta_file)
    
    # Check for classes directory (organized structure)
    classes_dir = base_path_obj / "classes"
    if not classes_dir.exists():
        result["issues"].append(f"ESC-50 classes directory not found: {classes_dir}")
    else:
        result["classes_dir"] = str(classes_dir)
    
    # If all essential components are available
    if result["audio_dir"] and result["meta_file"]:
        result["available"] = True
    
    return result

def get_esc10_files_from_metadata(meta_file: str) -> List[Dict]:
    """
    Extract ESC-10 file information from ESC-50 metadata.
    
    Args:
        meta_file: Path to the ESC-50 metadata CSV file
        
    Returns:
        List of dictionaries containing ESC-10 file information
    """
    df = pd.read_csv(meta_file)
    
    # Filter for ESC-10 files only
    esc10_df = df[df['esc10'] == True].copy()
    
    if esc10_df.empty:
        raise ValueError("No ESC-10 files found in metadata")
    
    # Convert to list of dictionaries for easier processing
    esc10_files = esc10_df.to_dict('records')
    
    print(f"Found {len(esc10_files)} ESC-10 files across {len(esc10_df['category'].unique())} categories")
    print(f"ESC-10 categories: {sorted(esc10_df['category'].unique())}")
    
    return esc10_files

def create_esc10_structure(esc10_files: List[Dict], 
                          source_audio_dir: str, 
                          target_dir: str,
                          use_organized_source: bool = True) -> str:
    """
    Create ESC-10 directory structure and copy files.
    
    Args:
        esc10_files: List of ESC-10 file information from metadata
        source_audio_dir: Source directory containing ESC-50 audio files
        target_dir: Target directory for ESC-10 dataset
        use_organized_source: Whether to use organized classes structure or flat audio structure
        
    Returns:
        Path to the created ESC-10 dataset
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Create classes directory structure
    classes_dir = target_path / "classes"
    classes_dir.mkdir(exist_ok=True)
    
    # Create metadata directory
    meta_dir = target_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    
    # Group files by category for organization
    category_files = {}
    for file_info in esc10_files:
        category = file_info['category']
        if category not in category_files:
            category_files[category] = []
        category_files[category].append(file_info)
    
    # Create category directories and copy files
    copied_files = []
    failed_files = []
    
    source_base = Path(source_audio_dir)
    
    print(f"Creating ESC-10 dataset structure in: {target_path}")
    
    for category, files in tqdm(category_files.items(), desc="Processing categories"):
        # Create category directory
        category_dir = classes_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for file_info in tqdm(files, desc=f"Copying {category} files", leave=False):
            filename = file_info['filename']
            
            # Determine source file path
            if use_organized_source:
                # Look in organized classes structure first
                source_file = source_base.parent / "classes" / category / filename
                if not source_file.exists():
                    # Fallback to flat audio structure
                    source_file = source_base / filename
            else:
                # Use flat audio structure
                source_file = source_base / filename
            
            # Target file path
            target_file = category_dir / filename
            
            # Copy file if source exists and target doesn't exist
            if source_file.exists():
                if not target_file.exists():
                    shutil.copy2(source_file, target_file)
                copied_files.append({
                    'filename': filename,
                    'category': category,
                    'source': str(source_file),
                    'target': str(target_file)
                })
            else:
                failed_files.append({
                    'filename': filename,
                    'category': category,
                    'attempted_source': str(source_file)
                })
    
    # Create ESC-10 specific metadata
    esc10_df = pd.DataFrame(esc10_files)
    esc10_meta_file = meta_dir / "esc10.csv"
    esc10_df.to_csv(esc10_meta_file, index=False)
    
    # Create summary file
    summary = {
        "total_esc10_files": len(esc10_files),
        "successfully_copied": len(copied_files),
        "failed_copies": len(failed_files),
        "categories": list(category_files.keys()),
        "files_per_category": {cat: len(files) for cat, files in category_files.items()}
    }
    
    print(f"\nESC-10 Dataset Creation Summary:")
    print(f"  Total ESC-10 files: {summary['total_esc10_files']}")
    print(f"  Successfully copied: {summary['successfully_copied']}")
    print(f"  Failed copies: {summary['failed_copies']}")
    print(f"  Categories: {len(summary['categories'])}")
    
    if failed_files:
        print(f"\nFailed files:")
        for failed in failed_files[:5]:  # Show first 5 failed files
            print(f"  - {failed['filename']} (category: {failed['category']})")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    return str(classes_dir)

def download_esc10() -> str:
    """
    Create ESC-10 dataset by extracting files from existing ESC-50 dataset.
    
    Returns:
        Path to the created ESC-10 dataset directory.
    """
    # Check ESC-50 availability
    esc50_status = check_esc50_availability()
    
    if not esc50_status["available"]:
        error_msg = "ESC-50 dataset is not available. Please download ESC-50 first.\n"
        error_msg += "Issues found:\n"
        for issue in esc50_status["issues"]:
            error_msg += f"  - {issue}\n"
        error_msg += "\nTo download ESC-50, run: python download_esc50.py"
        raise FileNotFoundError(error_msg)
    
    print("ESC-50 dataset found! Extracting ESC-10 subset...")
    
    # Get target directory for ESC-10
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    esc10_dir = repo_root / "src" / "datasets" / "ESC-10-master"
    
    # Load ESC-10 files from metadata
    esc10_files = get_esc10_files_from_metadata(esc50_status["meta_file"])
    
    # Create ESC-10 dataset
    dataset_path = create_esc10_structure(
        esc10_files=esc10_files,
        source_audio_dir=esc50_status["audio_dir"],
        target_dir=str(esc10_dir),
        use_organized_source=esc50_status["classes_dir"] is not None
    )
    
    print(f"\nESC-10 dataset successfully created!")
    print(f"Dataset location: {dataset_path}")
    
    return dataset_path

def main():
    """Main function to create ESC-10 dataset."""
    try:
        dataset_path = download_esc10()
        print(f"\nESC-10 dataset successfully created!")
        print(f"Dataset location: {dataset_path}")
        print(f"\nTo use this dataset, point your data_path to: {dataset_path}")
        
        # Verify the created dataset
        if Path(dataset_path).exists():
            categories = [d.name for d in Path(dataset_path).iterdir() if d.is_dir()]
            print(f"\nVerification: Found {len(categories)} categories in ESC-10:")
            for cat in sorted(categories):
                files = list((Path(dataset_path) / cat).glob("*.wav"))
                print(f"  - {cat}: {len(files)} files")
        
    except Exception as e:
        print(f"Error creating ESC-10 dataset: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 