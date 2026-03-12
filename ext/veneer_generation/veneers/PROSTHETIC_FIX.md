# Critical Fix: Prosthetic/Denture Look Elimination

## Problem Analysis

The model was generating **prosthetic-looking results** with anatomically incorrect outputs:
- Teeth appeared like dentures/dental prosthetics
- Unnatural lip replacement
- Loss of gum texture and depth
- Overly rigid, template-like geometry

### Root Causes Identified

1. **Mask Too Large**: Included gums, inner lips, and mouth cavity → diffusion replaced lip volume with prosthetic tissue
2. **Ideal Arch Too Strong**: Drew literal blueprint with strong intensities (200, 255) → ControlNet treated it as rigid enforcement
3. **ControlNet Scale Too High**: 0.65-0.97 → structural dictator mode instead of gentle guidance
4. **Denoising Strength Too High**: 0.88-0.97 → complete reconstruction instead of subtle geometry correction

## Fixes Applied

### Fix #1: Aggressive Mask Erosion (Shrink to Enamel Only)

**File**: [`mask_engine.py:175-179`](veneers/engine/mask_engine.py:175-179)

```python
# Additional aggressive erosion to pull mask inward from gum boundary
# This ensures we only modify enamel region, preserving natural gum and lip appearance
kernel_shrink = np.ones((5, 5), np.uint8)
smoothed = cv2.erode(smoothed, kernel_shrink, iterations=2)
```

**Impact**:
- Mask now covers **enamel ONLY**
- Excludes: gums, inner lips, mouth cavity
- Prevents lip volume replacement
- Preserves natural gum texture and depth

---

### Fix #2: Soften Ideal Arch Conditioning (Guidance, Not Blueprint)

**File**: [`conditioning_engine.py:98-149`](veneers/engine/conditioning_engine.py:98-149)

**Changes**:
- Tooth separators: `200 → 120` intensity
- Arch outline: `200 → 120` intensity, thickness `2 → 1`
- Midline: `150 → 100` intensity
- Occlusion plane: `255 → 180` intensity, thickness `2 → 1`
- Gaussian blur: `(5,5) σ=1 → (9,9) σ=2`

**Impact**:
- Ideal arch is now **subtle guidance**, not rigid template
- ControlNet gets gentle alignment hints, not literal blueprint
- Preserves original facial structure and lighting
- Allows diffusion creative freedom for natural results

---

### Fix #3: Rebalance All Presets (Lower ControlNet Scale & Strength)

**File**: [`presets.py:8-145`](veneers/presets.py:8-145)

#### Natural Preset
```python
"num_inference_steps": 32,        # +4 (better quality)
"guidance_scale": 8.5,            # -0.5 (less rigid)
"controlnet_conditioning_scale": 0.45,  # -0.20 (CRITICAL: allows lips/face to dominate)
"strength": 0.82,                 # -0.06 (geometry correction, not reconstruction)
```

#### Dramatic Preset
```python
"num_inference_steps": 36,
"guidance_scale": 9.5,            # -0.5
"controlnet_conditioning_scale": 0.52,  # -0.23 (still dramatic, not prosthetic)
"strength": 0.87,                 # -0.05 (preserve facial structure)
```

#### Conservative Preset
```python
"controlnet_conditioning_scale": 0.38,  # -0.17 (very subtle guidance)
"strength": 0.72,                 # -0.03 (minimal intervention)
```

#### Fast Preset
```python
"num_inference_steps": 20,        # +2
"controlnet_conditioning_scale": 0.45,  # -0.20
"strength": 0.80,                 # -0.05
```

#### Veneers Max Preset (Worst Offender)
```python
"num_inference_steps": 36,
"guidance_scale": 9.5,            # -1.5 (was WAY too high)
"controlnet_conditioning_scale": 0.48,  # -0.20 (was causing prosthetic look)
"strength": 0.85,                 # -0.12 (was complete replacement)
```

**Impact**:
- **ControlNet Scale 0.45-0.52**: Influence, not enforcement
- **Strength 0.72-0.87**: Subtle correction, not full reconstruction
- Preserves lips, gum tone, depth, and facial structure
- Straightens arch subtly without prosthetic appearance

---

### Fix #4: Diagnostic Mode (Edges-Only Conditioning)

**Files**:
- [`conditioning_engine.py:28-65`](veneers/engine/conditioning_engine.py:28-65)
- [`pipeline.py:54-77, 138-145`](veneers/pipeline.py:54-77)

```python
# In ConditioningEngine.build()
use_ideal_arch=True  # Set to False for diagnostic

# In Pipeline.run()
diagnostic_edges_only=False  # Set to True to test edges-only
```

**Usage**:
```python
# Test if ideal arch is too strong
result = pipeline.run(
    image="input.jpg",
    preset="natural",
    diagnostic_edges_only=True  # Uses edges only, no ideal arch
)
```

If edges-only produces good results → ideal arch conditioning was too strong (now fixed).

---

## The Tipping Point Concept

There's a critical threshold where ControlNet transitions from:

**✅ Structural Guide** → **❌ Structural Dictator**

### Previous State (Dictator Mode)
- ControlNet scale: 0.65-0.97
- Strong arch intensities: 200-255
- High denoising: 0.88-0.97
- Result: Rigid, prosthetic, template-like

### New State (Guide Mode)
- ControlNet scale: 0.45-0.52
- Soft arch intensities: 100-180
- Moderate denoising: 0.72-0.87
- Result: Natural, subtle, preserves facial structure

---

## Expected Behavior After Fixes

### ✅ What Should Happen

1. **Mask Coverage**: Enamel only (visible in `tooth_mask_binary.png`)
2. **Lip Preservation**: Lips maintain volume, texture, depth
3. **Gum Preservation**: Natural gum color and texture retained
4. **Gentle Alignment**: Teeth straighten subtly without prosthetic look
5. **Facial Structure**: Original lighting, depth, and facial features preserved
6. **Natural Veneers**: Look like enhanced real teeth, not dentures

### ❌ What Should NOT Happen

1. ~~Prosthetic/denture appearance~~
2. ~~Lip volume replacement~~
3. ~~Rigid, template-like geometry~~
4. ~~Loss of gum texture~~
5. ~~Unnatural depth/lighting~~
6. ~~Double-mouth artifacts~~

---

## Debug Checklist

When `debug_dir` is enabled, verify:

1. **`tooth_mask_binary.png`**:
   - Should show **enamel only** (small, tight mask)
   - Should NOT include gums or inner lips
   - Should be aggressively eroded inward

2. **`tooth_mask_feathered.png`**:
   - Should show smooth gradient for blending
   - Should maintain small coverage area

3. **`conditioning.png`**:
   - **Red channel**: Binary mask (enamel only)
   - **Green channel**: Softened ideal arch + edges (subtle lines, not strong blueprint)
   - **Blue channel**: Distance transform
   - Ideal arch should be barely visible (soft, blurred guidance)

4. **`output.png`**:
   - Natural teeth enhancement
   - Preserved lips and gums
   - Subtle straightening
   - No prosthetic appearance

---

## Parameter Tuning Guide

### If teeth are still too prosthetic:
```python
# Lower ControlNet influence even more
"controlnet_conditioning_scale": 0.38  # Down from 0.45

# Or reduce denoising strength
"strength": 0.75  # Down from 0.82
```

### If teeth aren't straightening enough:
```python
# Slightly increase ControlNet (but stay under 0.55!)
"controlnet_conditioning_scale": 0.48  # Up from 0.45

# Or increase denoising slightly
"strength": 0.85  # Up from 0.82
```

### If mask is still too big:
```python
# Increase erosion iterations in mask_engine.py line 178
smoothed = cv2.erode(smoothed, kernel_shrink, iterations=3)  # Was 2
```

### If ideal arch is still too strong:
```python
# Further reduce intensities in conditioning_engine.py
cv2.line(arch, pt_top, pt_bot, 80, 1)  # Down from 120
```

---

## Mathematical Insight

The key insight is **magnitude control**:

### Conditioning Strength Hierarchy
```
Mask Binary (255)           → Spatial boundary
  ↓
Edges Real (0-255)          → Structural detail
  ↓
Ideal Arch (80-180)         → Subtle guidance (NEW: softened)
  ↓
Distance Transform (0-255)  → Spatial awareness
```

### ControlNet Influence Curve
```
0.30-0.40: Subtle hint (conservative)
0.40-0.50: Gentle guidance (natural) ← NEW SWEET SPOT
0.50-0.60: Moderate influence (dramatic)
0.60-0.70: Strong enforcement (danger zone)
0.70+:     Dictator mode (prosthetic look) ← OLD RANGE
```

---

## Production Recommendations

### For Startup MVP
Use **"natural"** preset (now rebalanced):
```python
result = pipeline.run(image="input.jpg", preset="natural")
```

### For Clinical-Grade SaaS
Implement **adaptive preset selection**:
```python
def select_preset(mask_analysis):
    asymmetry_score = analyze_tooth_asymmetry(mask)

    if asymmetry_score < 0.2:
        return "conservative"  # Already good teeth
    elif asymmetry_score < 0.5:
        return "natural"  # Normal case
    else:
        return "dramatic"  # Significant correction needed
```

### For Advanced Users
Expose **ControlNet scale slider**:
- Range: 0.30-0.55
- Default: 0.45
- Label: "Alignment Strength" (not "ControlNet Scale")

---

## Testing Protocol

1. **Run with natural preset**:
```bash
python -m veneers.cli \
  --controlnet ./weights/controlnet \
  --image test.jpg \
  --output output.jpg \
  --preset natural \
  --debug-dir ./debug
```

2. **Check debug outputs** (see Debug Checklist above)

3. **If still prosthetic**, run diagnostic:
```python
result = pipeline.run(
    image="test.jpg",
    preset="natural",
    diagnostic_edges_only=True
)
```

4. **Compare results**:
   - If edges-only looks good → ideal arch still too strong
   - If edges-only also looks prosthetic → ControlNet scale too high

---

## Files Modified

1. **`veneers/engine/mask_engine.py`**: Added aggressive erosion (2 iterations, 5x5 kernel)
2. **`veneers/engine/conditioning_engine.py`**: Softened all arch intensities, increased blur
3. **`veneers/presets.py`**: Rebalanced all 5 presets (lower scale & strength)
4. **`veneers/pipeline.py`**: Added diagnostic_edges_only parameter

---

## Next-Level Enhancements (Future)

### 1. Auto-Scale Conditioning Based on Mask Size
```python
mask_area = np.sum(mask_binary > 0)
scale_factor = np.clip(mask_area / 50000, 0.3, 0.5)
controlnet_conditioning_scale = scale_factor
```

### 2. Mathematically Smooth Geometry Encoding
Replace line-based arch with spline-based smooth curves using `scipy.interpolate`.

### 3. GAN-Based Tooth Realism Enhancement
Post-process generated teeth with StyleGAN for photorealistic enamel texture.

### 4. Automatic Occlusion Detection → Preset Escalation
Detect severe malocclusion → automatically use higher strength preset.

---

## Conclusion

The fixes transform the system from **structural dictator** to **gentle guide**:

- **Mask**: Enamel only (not gums/lips)
- **Arch**: Soft guidance (not blueprint)
- **ControlNet**: Influence (not enforcement)
- **Strength**: Correction (not reconstruction)

Result: **Natural veneer enhancement**, not prosthetic replacement.
