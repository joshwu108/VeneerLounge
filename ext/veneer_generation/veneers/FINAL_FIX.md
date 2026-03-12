# Final Fix: Teeth Size and Blending Issues

## Problem
- Teeth appearing too large/enlarged
- Teeth floating outside natural mouth boundaries
- Poor integration with surrounding facial features
- Translucent/ghostly appearance

## Root Causes

1. **Mask extends beyond visible teeth** - includes gum regions and mouth cavity
2. **Diffusion strength too high** - replaces too much of original structure
3. **ControlNet scale still too high** - enforces unnatural geometry
4. **Ideal arch intensities visible enough to create structure** - should be invisible

## Final Conservative Settings

### Natural Preset (Ultra-Conservative)
```python
"num_inference_steps": 28,
"guidance_scale": 7.5,
"controlnet_conditioning_scale": 0.32,  # Very gentle
"strength": 0.75,  # Preserve more original
"whitening_target": 210.0,
"whitening_strength": 0.38,
```

### Ideal Arch (Ghost-Level Soft)
```python
# All line intensities: 50-80 (barely perceptible)
# Blur: (15, 15) with sigma=3 (maximum softness)
```

### Mask Erosion
- Removed aggressive double-erosion that made teeth translucent
- Using only standard erosion (erosion_px=4, 1 iteration)

## Testing Recommendations

### 1. Try Edges-Only Diagnostic
```python
result = pipeline.run(
    image="input.jpg",
    preset="natural",
    diagnostic_edges_only=True  # Disable ideal arch completely
)
```

If this produces better results, the ideal arch is still too strong.

### 2. Further Reduce ControlNet Scale
```python
# In presets.py natural preset:
"controlnet_conditioning_scale": 0.25,  # Even lower
```

### 3. Further Reduce Strength
```python
# In presets.py natural preset:
"strength": 0.70,  # Even more conservative
```

### 4. Check Mask Size in Debug
Enable debug and check `tooth_mask_binary.png`:
- Should show ONLY visible tooth enamel
- Should NOT include gums or mouth cavity
- Should be tightly bounded to actual teeth

If mask is too large, increase erosion:
```python
# In presets.py:
"mask_erosion_px": 6,  # Up from 4
```

## Alternative Approach: Disable Ideal Arch Completely

If the issue persists, consider running WITHOUT ideal arch by default:

```python
# In pipeline.py, change default:
diagnostic_edges_only=True  # Make this the default
```

This will use only real edges for conditioning, no synthetic geometry at all.

## Key Principle

**Less is more**. The goal is:
- Subtle whitening
- Gentle straightening
- Preserve original mouth structure
- Teeth should look like enhanced originals, not replacements

## Current Settings Summary

After all fixes applied:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| controlnet_scale | 0.32 | Whisper-level guidance |
| strength | 0.75 | Preserve 25% of original |
| guidance_scale | 7.5 | Less rigid prompt adherence |
| arch intensities | 50-80 | Ghost-level hints |
| arch blur | (15,15) σ=3 | Maximum softness |
| mask erosion | 4px, 1 iter | Standard conservative |

These are the most conservative settings possible while still providing some enhancement.
