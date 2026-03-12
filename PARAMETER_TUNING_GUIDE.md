# Veneer Pipeline Parameter Tuning Guide

## The Core Trade-off

The veneer generation pipeline has **two competing objectives**:

1. **Preserve natural tooth structure** (avoid distorted/fake-looking teeth)
2. **Transform teeth enough to look like veneers** (whitening, alignment, smoothing)

### Key Parameters

#### `controlnet_conditioning_scale` (0.0-1.0)
**What it does**: Controls how strictly the diffusion follows the conditioning image (edges + ideal arch)

- **0.9-1.0**: Locks to original structure - NO geometry change, teeth stay exactly as they are
- **0.7-0.8**: Very conservative - minimal reshaping, mostly just texture/color changes
- **0.55-0.65**: **BALANCED** - allows visible reshaping while preserving natural tooth proportions
- **0.4-0.5**: Aggressive - significant geometry freedom, risk of unnatural appearance
- **0.2-0.3**: Extreme - can distort face structure, teeth may look prosthetic

#### `strength` (0.0-1.0)
**What it does**: Determines how much of the diffusion process to run (denoising ratio)

- **0.3-0.5**: Minimal change - cosmetic only (slight whitening/smoothing)
- **0.6-0.7**: Moderate change - visible but subtle transformation
- **0.75-0.85**: **STRONG** - clear veneer effect, noticeable whitening and smoothing
- **0.88-0.95**: Very strong - maximum transformation, risk of over-processing

#### `num_inference_steps` (10-50)
**What it does**: Quality of the diffusion process

- Actual denoising steps = `num_inference_steps * strength`
- Example: `steps=30, strength=0.80` → 24 actual denoising steps
- **Minimum recommended**: 25 steps
- **Balanced**: 30-35 steps
- **High quality**: 40+ steps

#### `guidance_scale` (5.0-12.0)
**What it does**: How strongly to follow the text prompt

- **5.0-6.5**: Loose prompt following, more natural variation
- **7.0-8.5**: **BALANCED** - good prompt adherence without artifacts
- **9.0-10.5**: Strong prompt following, sharper details, risk of over-sharpening
- **11.0+**: Very strong, can create artifacts and unrealistic gloss

---

## Current Preset Configurations

### conservative (for subtle enhancement)
```python
controlnet = 0.55  # Medium structure preservation
strength = 0.82    # Strong enough for visible change
steps = 30         # Good quality
guidance = 7.5     # Balanced
```
**Use when**: Client wants natural-looking whitening with minimal geometry change

**Expected result**: Teeth 15-25% whiter, slight smoothing, maintains original tooth shapes

---

### natural (recommended default)
```python
controlnet = 0.58  # Balanced structure vs. transformation
strength = 0.85    # Strong transformation
steps = 35         # High quality
guidance = 8.0     # Good prompt adherence
```
**Use when**: Standard veneer preview - realistic but noticeable improvement

**Expected result**: Teeth 25-35% whiter, visible smoothing, subtle alignment correction

---

### dramatic (maximum cosmetic effect)
```python
controlnet = 0.48  # More geometry freedom
strength = 0.90    # Very strong transformation
steps = 40         # Maximum quality
guidance = 9.0     # Strong prompt following
```
**Use when**: Client wants "Hollywood smile" - maximum whitening and perfect alignment

**Expected result**: Teeth 35-45% whiter, significant smoothing, visible alignment improvement

---

### test_extreme (diagnostic only)
```python
controlnet = 0.25  # Minimal structure preservation
strength = 0.95    # Maximum denoising
steps = 40         # High quality
guidance = 9.0     # Strong guidance
```
**Use when**: Testing if the pipeline can make ANY change at all

**Warning**: May produce artificial-looking teeth. For diagnostics only.

---

## Troubleshooting

### Problem: Teeth look exactly the same (no visible change)

**Diagnosis**: Diffusion delta < 5.0

**Solutions** (try in order):
1. Reduce `controlnet_conditioning_scale` by 0.05-0.10
2. Increase `strength` by 0.05-0.10
3. Check mask quality (should be 10-35% coverage)
4. Try `test_extreme` preset to verify pipeline works at all

---

### Problem: Teeth look distorted/fake/prosthetic

**Diagnosis**: Diffusion delta > 30.0, OR unnatural tooth shapes

**Solutions** (try in order):
1. **Increase** `controlnet_conditioning_scale` by 0.05-0.10
2. Reduce `strength` by 0.05
3. Reduce `guidance_scale` by 0.5-1.0
4. Check if ideal_arch conditioning is too aggressive
5. Try conservative preset

---

### Problem: Teeth are too white/glowing/blue-tinted

**Diagnosis**: Post-processing whitening too aggressive

**Solutions**:
1. Reduce `whitening_target` by 5.0 (e.g., 215 → 210)
2. Reduce `whitening_strength` by 0.10
3. Check LAB color space values in debug output

---

### Problem: Mask includes lips/gums

**Diagnosis**: Mask coverage > 40%

**Solutions**:
1. Tighten bounding box (reduce height of crop region)
2. Increase `mask_erosion_px` from 3 to 4-5
3. Check segmentation model is loaded correctly
4. Review `03_binary_mask.png` in debug output

---

### Problem: Mask is too small/empty

**Diagnosis**: Mask coverage < 5%

**Solutions**:
1. Check input image quality (teeth visible, well-lit)
2. Adjust bounding box to focus on teeth region
3. Review `02_raw_mask.png` and `03_binary_mask.png`
4. If segmentation fails, check if intelligent rectangle fallback runs

---

## Diagnostic Workflow

### Step 1: Check mask quality
```bash
# Look at debug outputs
open debug_outputs/run_TIMESTAMP/02_raw_mask.png
open debug_outputs/run_TIMESTAMP/03_binary_mask.png
```

**Target**: 10-35% coverage, mask clearly outlines teeth only

---

### Step 2: Check diffusion output
```bash
open debug_outputs/run_TIMESTAMP/08_raw_diffusion_output.png
open debug_outputs/run_TIMESTAMP/15_diffusion_delta.png
```

**Target**: Visible changes in delta image (bright regions = changed pixels)

---

### Step 3: Measure diffusion delta
```python
/path/to/venv/bin/python3 << 'EOF'
from PIL import Image
import numpy as np

orig = np.array(Image.open('debug_outputs/run_TIMESTAMP/01_original.png'))
diff = np.array(Image.open('debug_outputs/run_TIMESTAMP/08_raw_diffusion_output.png'))
mask = np.array(Image.open('debug_outputs/run_TIMESTAMP/03_binary_mask.png')) > 127

delta = np.abs(orig[mask].astype(float) - diff[mask].astype(float)).mean()
print(f"Diffusion delta: {delta:.2f}")

if delta < 5.0:
    print("⚠️ Too low - reduce controlnet or increase strength")
elif delta < 10.0:
    print("⚠️ Borderline - consider reducing controlnet slightly")
elif delta > 40.0:
    print("⚠️ Too high - increase controlnet or reduce strength")
else:
    print("✅ Good range")
EOF
```

---

## Parameter Adjustment Recipe

### To increase transformation (if teeth barely change):

```python
# Option 1: Reduce ControlNet (more geometry freedom)
controlnet_conditioning_scale -= 0.08

# Option 2: Increase strength (more denoising)
strength += 0.08

# Option 3: Both (aggressive)
controlnet_conditioning_scale -= 0.05
strength += 0.05
```

### To reduce artifacts (if teeth look fake):

```python
# Option 1: Increase ControlNet (preserve structure)
controlnet_conditioning_scale += 0.08

# Option 2: Reduce strength (less modification)
strength -= 0.05

# Option 3: Reduce guidance (softer prompt following)
guidance_scale -= 1.0
```

---

## Advanced: Two-Pass Mode

For maximum quality with geometry change, use two-pass mode:

```python
result = pipeline.run(
    image=image,
    preset="natural",
    two_pass=True
)
```

**Pass 1** (Geometry Reshape):
- Low controlnet (0.35), high strength (0.88)
- Reshapes tooth geometry toward ideal arch

**Pass 2** (Texture Refinement):
- Medium controlnet (0.55), medium strength (0.70)
- Refines surface texture and porcelain appearance

**When to use**: Complex cases with significant misalignment

**Warning**: Takes ~2x longer (two full diffusion runs)

---

## Understanding the Metrics

### Mask Coverage
- **< 5%**: Teeth not detected (segmentation failure)
- **10-20%**: Good - teeth only
- **20-35%**: Acceptable - may include some gum line
- **35-50%**: Warning - likely includes lips
- **> 50%**: Failure - segmentation broken

### Diffusion Delta (mean absolute pixel difference in mask)
- **< 3.0**: No effective change (too constrained)
- **3.0-5.0**: Minimal change (cosmetic only)
- **8.0-15.0**: ✅ **Visible transformation** (veneer effect)
- **15.0-25.0**: Strong transformation
- **> 30.0**: Very aggressive (may look unnatural)

### LAB L Channel (lightness in LAB color space)
- **Input**: 140-180 (natural tooth color)
- **Target**: 210-220 (veneer white)
- **> 230**: Too bright (glowing/bloom effect)

---

## Quick Reference: Parameter Ranges

| Parameter | Conservative | Natural | Dramatic |
|-----------|--------------|---------|----------|
| controlnet_scale | 0.55-0.60 | 0.55-0.58 | 0.45-0.50 |
| strength | 0.80-0.85 | 0.85-0.88 | 0.88-0.92 |
| steps | 25-30 | 30-35 | 35-40 |
| guidance | 7.0-7.5 | 7.5-8.5 | 8.5-9.5 |
| whitening_target | 210-212 | 213-216 | 216-220 |
| whitening_strength | 0.35-0.40 | 0.42-0.48 | 0.50-0.55 |

---

## Testing Checklist

When tuning parameters:

- [ ] Check mask coverage (should be 10-35%)
- [ ] Check diffusion delta (should be 8-25 for visible change)
- [ ] Check for lip/gum inclusion in mask
- [ ] Verify actual denoising steps = `steps * strength`
- [ ] Review debug images for visual quality
- [ ] Test with multiple patient images (different lighting, angles)
- [ ] Check for blue tint (LAB b-channel < 120)
- [ ] Verify natural tooth proportions preserved
- [ ] Compare before/after side-by-side

---

## Key Insights

1. **ControlNet is NOT "quality"** - it's **structure preservation**
   - High ControlNet = preserve original (less transformation)
   - Low ControlNet = allow change (more transformation)

2. **Strength determines power, ControlNet determines direction**
   - Think of it like: strength = "how hard to push", controlnet = "stay in lane"

3. **The sweet spot changes per image**
   - Crooked teeth need lower controlnet
   - Already-nice teeth need higher controlnet
   - Auto-adjustment helps but manual tuning may be needed

4. **More steps ≠ more change**
   - More steps = smoother, higher quality
   - Change amount comes from strength and controlnet

5. **Prompt weight ceiling**
   - Beyond guidance=9.0, benefits diminish
   - High guidance can create gloss artifacts

---

## Files Modified

- [ext/veneer_generation/veneers/presets.py](ext/veneer_generation/veneers/presets.py)
- [ext/veneer_generation/veneers/engine/mask_engine.py](ext/veneer_generation/veneers/engine/mask_engine.py)
- [ext/veneer_generation/veneers/pipeline.py](ext/veneer_generation/veneers/pipeline.py)
