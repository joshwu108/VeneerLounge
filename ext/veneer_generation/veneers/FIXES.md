# Critical Fixes Applied

## Issue: Double-Mouth Artifact & Poor Generation Quality

### Root Cause
The new modular architecture had a mask feathering bug that caused:
1. Feathering was applied during mask generation, converting binary mask → gradient mask
2. Pipeline's `if mask_np.mean() < 10` check was evaluating a gradient mask instead of binary mask
3. Gradient masks have high mean values (lots of light gray pixels), failing the check
4. Fallback rectangle mask was being used instead of actual tooth segmentation
5. Ideal arch geometry was drawn on the rectangle, not on real teeth
6. Diffusion generated teeth in wrong location → double-mouth artifact

### Fix Applied

**Separated binary and feathered masks**:
1. `MaskEngine.generate()` now returns binary mask (no feathering)
2. Added `MaskEngine.feather()` as separate method
3. Pipeline now:
   - Generates binary mask
   - Checks validity on binary mask (`mean < 10` now makes sense)
   - Feathers mask separately
   - Uses **binary mask for conditioning** (ideal arch)
   - Uses **feathered mask for diffusion/blending** (smooth compositing)

## Files Modified

### `/ext/veneer_generation/veneers/engine/mask_engine.py`
- Removed `self._feather(mask)` from `generate()` method
- Added public `feather()` method for explicit feathering
- Updated docstrings to clarify binary vs gradient masks

### `/ext/veneer_generation/veneers/pipeline.py`
- Split mask generation into two steps:
  - Step 1: Generate binary mask
  - Step 1b: Feather mask separately
- Use binary mask for:
  - Validity checking
  - Conditioning generation (ideal arch)
  - Post-processing (whitening)
- Use feathered mask for:
  - Diffusion inpainting
  - Final blending
- Added debug output for both binary and feathered masks

## Expected Behavior After Fix

1. ✅ Tooth segmentation works correctly
2. ✅ Ideal arch is drawn on actual teeth (not fallback rectangle)
3. ✅ Diffusion generates teeth in correct location
4. ✅ No double-mouth artifacts
5. ✅ Smooth blending with feathered mask
6. ✅ Precise whitening with binary mask

## Debug Output

When `debug_dir` is set, you'll now see:
- `tooth_mask_binary.png` - Sharp binary mask showing actual teeth
- `tooth_mask_feathered.png` - Gradient mask for smooth blending
- `conditioning.png` - Should show ideal arch on teeth (not rectangle)
- `input_crop.png` - Cropped input image
- `output.png` - Final result

## Testing

Run the pipeline with debug enabled:
```python
from veneers import VeneerPipeline

pipeline = VeneerPipeline(
    controlnet_path="./weights/controlnet",
    segmentation_checkpoint="./weights/seg.pth"
)

result = pipeline.run(
    image="input.jpg",
    preset="natural",
    debug_dir="./debug"
)
```

Check `debug/tooth_mask_binary.png` - should show clear white teeth on black background, not a white rectangle.
