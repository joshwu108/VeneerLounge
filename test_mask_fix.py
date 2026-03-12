#!/usr/bin/env python3
"""
Test script to verify mask inversion fix.
"""
import sys
from pathlib import Path

# Add veneer_generation to path
sys.path.insert(0, str(Path(__file__).parent / 'ext' / 'veneer_generation'))

from veneers import VeneerPipeline
from PIL import Image
import numpy as np

# Test image
test_image_path = Path(__file__).parent / 'debug_outputs' / 'input_image.jpg'
debug_dir = Path(__file__).parent / 'debug_outputs'

print("=" * 60)
print("Testing Veneer Pipeline with Mask Inversion Fix")
print("=" * 60)

# Initialize pipeline
print("\n1. Initializing pipeline...")
pipeline = VeneerPipeline(
    controlnet_path='lllyasviel/control_v11p_sd15_seg',
    base_model='runwayml/stable-diffusion-v1-5',
    segmentation_checkpoint=str(Path(__file__).parent / 'ext' / 'individual_tooth_segmentation' / 'checkpoints' / 'CP_teeth_seg.pth'),
    device='cuda'
)

# Load test image
print(f"\n2. Loading test image from {test_image_path}...")
image = Image.open(test_image_path).convert('RGB')

# Run pipeline with debug mode
print("\n3. Running veneer generation with 'dramatic' preset and debug mode...")
print("   This will save debug images to check mask quality")
print()

result = pipeline.run(
    image=image,
    preset='dramatic',
    debug_dir=str(debug_dir),
    seed=42
)

print("\n" + "=" * 60)
print("Test Completed!")
print("=" * 60)
print(f"\nCheck debug images in: {debug_dir}/run_<TIMESTAMP>/")
print("Key files to review:")
print("  - 02_raw_mask.png : Raw segmentation from neural model")
print("  - 03_binary_mask.png : Should show WHITE on teeth (not corners!)")
print("  - 15_diffusion_delta.png : Should show visible changes on teeth")
print("  - 11_final_result.png : Final output with veneer effect")
print()

# Save result
output_path = '/tmp/veneer_test_result.jpg'
result.save(output_path, quality=95)
print(f"Result saved to: {output_path}")
