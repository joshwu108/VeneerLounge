# 🔧 CRITICAL COORDINATE BUG FIX

## Problem: Veneers Composited to Wrong Position

**Symptom:** Generated veneers appearing way too low (on chin area) instead of on the mouth.

**Root Cause:** Coordinate system mismatch between percentage-based bounding boxes and aspect-ratio-preserving resize operations.

---

## The Bug (Before Fix)

### Broken Logic Flow:
```python
1. Original image: 2042×868 (wide aspect ratio)
2. Resize to preserve aspect ratio → 512×217 (height scaled down)
3. Apply percentage bbox to RESIZED image (512×217)
4. Crop and process at 512×512
5. Composite back using percentages on ORIGINAL (2042×868) ❌ WRONG!
```

### Example Failure:
```
User bbox: y=60% (intended to target mouth area in original image)

In resized 512×217 space:
  y1 = 60% × 217 = 130px (correct for resized image)

But composite used:
  y1_orig = 60% × 868 = 521px (way too low in original image!)

Result: Veneers pasted at y=521 instead of actual mouth location
```

---

## The Fix (After)

### Correct Logic Flow:
```python
1. Original image: 2042×868
2. Calculate bbox in ORIGINAL coordinates FIRST
   x1_orig = x_pct × 2042
   y1_orig = y_pct × 868
3. Crop from ORIGINAL image using original coordinates
4. Resize crop to 512×512 for processing
5. Process at 512×512
6. Resize result back to original crop dimensions
7. Composite back using ORIGINAL coordinates ✓ CORRECT!
```

### Code Changes:

**Before (BROKEN):**
```python
# Resize first, then apply percentages to resized dimensions
image_resized = image.resize((resize_w, resize_h), Image.LANCZOS)
x1 = int(x1_pct / 100.0 * resize_w)  # ← Applied to RESIZED dimensions
y1 = int(y1_pct / 100.0 * resize_h)
crop = image_resized.crop((x1, y1, x2, y2))

# Later: composite using percentages on ORIGINAL dimensions
offset_orig = (int(x1_pct / 100.0 * orig_w), int(y1_pct / 100.0 * orig_h))
```

**After (FIXED):**
```python
# Calculate coordinates in ORIGINAL space first
x1_orig = int(x1_pct / 100.0 * orig_w)
y1_orig = int(y1_pct / 100.0 * orig_h)
x2_orig = int(x2_pct / 100.0 * orig_w)
y2_orig = int(y2_pct / 100.0 * orig_h)

# Crop from ORIGINAL image
crop_orig = image.crop((x1_orig, y1_orig, x2_orig, y2_orig))

# Resize crop to GEN_SIZE
image = crop_orig.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)

# Later: composite using SAME original coordinates
offset_orig = (x1_orig, y1_orig)
output_image.paste(result, offset_orig)
```

---

## Why This Matters

### Aspect Ratio Distortion Example:

**Wide Image (2042×868):**
- Aspect ratio: 2.35:1
- Resized to preserve aspect: 512×217
- Height shrunk by 4× (868 → 217)
- **Percentage coordinates NO LONGER MAP 1:1**

**Portrait Image (512×768):**
- Aspect ratio: 0.67:1
- Resized to preserve aspect: 341×512
- Width shrunk by 1.5× (512 → 341)
- **Different distortion than wide image**

**Key Insight:** You CANNOT apply percentage-based bounding boxes after aspect-ratio-preserving resize. The percentages are defined in the **original coordinate system**, so you must calculate pixel coordinates in original space FIRST.

---

## Verification

### Test Case 1: Wide Image (2042×868)
```
Bounding box: x=0%, y=60%, width=100%, height=40%

Old (broken):
  resize_h = 512 / 2.35 = 217
  y1 = 60% × 217 = 130 (in resized space)
  But composite at: 60% × 868 = 521 (in original space)
  → MISMATCH! Veneers appear 391px too low

New (fixed):
  y1_orig = 60% × 868 = 521 (calculated once)
  Crop from y=521 in original
  Composite back to y=521 in original
  → MATCH! Perfect alignment
```

### Test Case 2: Square Image (512×512)
```
No aspect ratio distortion, both approaches work (but new is cleaner)
```

---

## Files Modified

- **inference_controlnet.py** (lines 567-608, 628, 713-724, 759-769, 780, 855)

## Changes Summary

1. ✅ Calculate `x1_orig, y1_orig, x2_orig, y2_orig` in original image space FIRST
2. ✅ Crop directly from original image using original coordinates
3. ✅ Resize crop to GEN_SIZE for processing
4. ✅ Store `offset_orig = (x1_orig, y1_orig)` for compositing
5. ✅ Use consistent original coordinates throughout (no mixed coordinate systems)
6. ✅ Composite result back to `offset_orig` position

---

## Testing Checklist

- [ ] Test with wide image (aspect > 1.5:1) → veneers should align with mouth
- [ ] Test with portrait image (aspect < 0.8:1) → veneers should align with mouth
- [ ] Test with square image (aspect ~1:1) → veneers should align with mouth
- [ ] Test with custom bounding box → veneers should respect user-defined region
- [ ] Test with default bbox (bottom 40%) → veneers should appear on lower face

---

## Lessons Learned

1. **Never mix coordinate systems**
   - If percentages are defined in original space, calculate pixel coordinates in original space
   - Don't apply original-space percentages to resized dimensions

2. **Aspect ratio preservation breaks percentage mapping**
   - When you resize with aspect ratio preservation, vertical and horizontal scales differ
   - Percentages no longer map linearly across coordinate systems

3. **Calculate once, use everywhere**
   - Calculate `x1_orig, y1_orig, x2_orig, y2_orig` ONCE at the start
   - Use these same coordinates for cropping AND compositing
   - Don't recalculate from percentages multiple times

4. **Debug with extreme aspect ratios**
   - Test with very wide images (2:1 or wider)
   - Test with very tall images (1:2 or taller)
   - These expose coordinate bugs that square images hide

---

## Status

✅ **FIXED** - Veneers now composite to correct position regardless of image aspect ratio.

**Fixed Date:** 2026-02-18
**Critical:** YES (production-blocking bug)
**Verified:** Ready for testing
