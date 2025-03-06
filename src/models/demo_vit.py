#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for testing ViT model with audio spectrograms.
This simplified test bed helps diagnose and fix issues with the ViT model processing.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from typing import Optional, Tuple, Any

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = "./cache"
NUM_CLASSES = 10  # Adjust based on your actual number of classes
SAMPLE_RATE = 16000
AUDIO_DURATION = 5  # seconds

def load_audio_sample(audio_path: str, target_sr: int = SAMPLE_RATE, target_duration: int = AUDIO_DURATION):
    """
    Load an audio sample and preprocess it.
    """
    logger.info(f"Loading audio from {audio_path}")
    
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Ensure consistent length
    target_length = target_sr * target_duration
    if len(audio) < target_length:
        # Pad if too short
        padding_length = target_length - len(audio)
        audio = np.pad(audio, (0, padding_length), mode='constant')
    else:
        # Trim if too long
        audio = audio[:target_length]
    
    logger.info(f"Audio shape after preprocessing: {audio.shape}")
    return audio

def create_spectrogram(audio_np: np.ndarray, sr: int = SAMPLE_RATE):
    """
    Convert audio to mel spectrogram.
    """
    logger.info("Creating mel spectrogram...")
    
    # Convert audio to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_np, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    logger.info(f"Mel spectrogram shape: {mel_spec_normalized.shape}")
    return mel_spec_normalized

def process_for_vit_1channel(mel_spec_normalized: np.ndarray):
    """
    Process the mel spectrogram for ViT model (1-channel approach).
    """
    logger.info("Processing for ViT (1-channel approach)...")
    
    # Convert normalized mel spectrogram to 8-bit image array
    image_array = (mel_spec_normalized * 255).astype(np.uint8)
    logger.info(f"Image array shape: {image_array.shape}")
    
    # Create PIL image
    pil_img = Image.fromarray(image_array, mode="L")
    logger.info(f"PIL image size: {pil_img.size}")
    
    # Load ViT processor
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224", cache_dir=CACHE_DIR)
    
    # Process image
    features = processor(pil_img, return_tensors="pt")
    logger.info(f"Features shape from processor: {features.pixel_values.shape}")
    
    # This will be a 1-channel tensor
    tensor_1ch = features.pixel_values[0]
    logger.info(f"1-channel tensor shape: {tensor_1ch.shape}")
    
    return tensor_1ch, processor

def process_for_vit_3channel(mel_spec_normalized: np.ndarray):
    """
    Process the mel spectrogram for ViT model (3-channel approach).
    """
    logger.info("Processing for ViT (3-channel approach)...")
    
    # Convert normalized mel spectrogram to 8-bit image array
    image_array = (mel_spec_normalized * 255).astype(np.uint8)
    
    # Create PIL image
    pil_img = Image.fromarray(image_array, mode="L")
    
    # Load ViT processor
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224", cache_dir=CACHE_DIR)
    
    # Process image
    features = processor(pil_img, return_tensors="pt")
    tensor_1ch = features.pixel_values[0]
    
    # Duplicate the tensor across the channel dimension to simulate RGB input
    tensor_3ch = tensor_1ch.repeat(3, 1, 1)  # Shape: [3, H, W]
    logger.info(f"3-channel tensor shape: {tensor_3ch.shape}")
    
    return tensor_3ch, processor

def create_vit_model(num_classes: int, processor: ViTImageProcessor):
    """
    Create a ViT model for image classification.
    """
    logger.info(f"Creating ViT model with {num_classes} classes...")
    
    model = ViTForImageClassification.from_pretrained(
        "google/vit-large-patch16-224",
        num_labels=num_classes,
        cache_dir=CACHE_DIR,
        ignore_mismatched_sizes=True
    )
    
    logger.info(f"Model created successfully")
    return model

def test_vit_with_1channel(audio_path: str):
    """
    Test ViT model with 1-channel input (will fail).
    """
    logger.info("=== Testing ViT with 1-channel input (expected to fail) ===")
    
    # Load and process audio
    audio = load_audio_sample(audio_path)
    mel_spec = create_spectrogram(audio)
    tensor_1ch, processor = process_for_vit_1channel(mel_spec)
    
    # Create model
    model = create_vit_model(NUM_CLASSES, processor)
    
    # Try to run inference (this will fail)
    try:
        logger.info("Attempting inference with 1-channel input...")
        with torch.no_grad():
            outputs = model(pixel_values=tensor_1ch.unsqueeze(0))
        logger.info("Inference successful (unexpected!)")
        logger.info(f"Output shape: {outputs.logits.shape}")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        logger.info("This error is expected - the model requires 3-channel input")

def test_vit_with_3channel(audio_path: str):
    """
    Test ViT model with 3-channel input (should work).
    """
    logger.info("=== Testing ViT with 3-channel input (should succeed) ===")
    
    # Load and process audio
    audio = load_audio_sample(audio_path)
    mel_spec = create_spectrogram(audio)
    tensor_3ch, processor = process_for_vit_3channel(mel_spec)
    
    # Create model
    model = create_vit_model(NUM_CLASSES, processor)
    
    # Run inference
    try:
        logger.info("Attempting inference with 3-channel input...")
        with torch.no_grad():
            outputs = model(pixel_values=tensor_3ch.unsqueeze(0))
        logger.info("Inference successful!")
        logger.info(f"Output shape: {outputs.logits.shape}")
        logger.info(f"Output logits: {outputs.logits}")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        logger.error("This is unexpected - the 3-channel approach should work")

def modify_vit_for_1channel():
    """
    Modify ViT model to accept 1-channel input (alternative approach).
    """
    logger.info("=== Modifying ViT to accept 1-channel input ===")
    
    model = ViTForImageClassification.from_pretrained(
        "google/vit-large-patch16-224",
        num_labels=NUM_CLASSES,
        cache_dir=CACHE_DIR
    )
    
    # Get the original projection layer
    original_projection = model.vit.embeddings.patch_embeddings.projection
    logger.info(f"Original projection layer: {original_projection}")
    
    # Modify the projection layer to accept 1-channel input
    model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
        in_channels=1,  # Change from 3 to 1 for grayscale
        out_channels=original_projection.out_channels,
        kernel_size=original_projection.kernel_size,
        stride=original_projection.stride,
        padding=original_projection.padding
    )
    
    # Initialize the weights of the new projection layer
    with torch.no_grad():
        original_weights = original_projection.weight.data
        new_weights = original_weights.mean(dim=1, keepdim=True)
        model.vit.embeddings.patch_embeddings.projection.weight.data = new_weights
    
    # Update the model configuration
    model.config.num_channels = 1
    
    logger.info(f"Modified projection layer: {model.vit.embeddings.patch_embeddings.projection}")
    logger.info("Model modified to accept 1-channel input")
    
    return model

def test_modified_vit(audio_path: str):
    """
    Test the modified ViT model with 1-channel input.
    """
    logger.info("=== Testing modified ViT with 1-channel input ===")
    
    # Load and process audio
    audio = load_audio_sample(audio_path)
    mel_spec = create_spectrogram(audio)
    tensor_1ch, processor = process_for_vit_1channel(mel_spec)
    
    # Create and modify model
    model = modify_vit_for_1channel()
    
    # Run inference
    try:
        logger.info("Attempting inference with 1-channel input on modified model...")
        with torch.no_grad():
            outputs = model(pixel_values=tensor_1ch.unsqueeze(0))
        logger.info("Inference successful!")
        logger.info(f"Output shape: {outputs.logits.shape}")
        logger.info(f"Output logits: {outputs.logits}")
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")

def main():
    """
    Main function to run the demo.
    """
    # Replace with the path to your audio file
    audio_path = "path/to/your/audio/file.wav"
    
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        logger.info("Please provide a valid audio file path when running the script.")
        return
    
    # Run tests
    test_vit_with_1channel(audio_path)
    print("\n" + "="*80 + "\n")
    test_vit_with_3channel(audio_path)
    print("\n" + "="*80 + "\n")
    test_modified_vit(audio_path)

if __name__ == "__main__":
    main()
