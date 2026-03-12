# 🔧 Default Bounding Box Position Fix

## Problem
Veneers appearing way too low (on chin/neck area) when no bounding box is provided by the frontend.

## Root Cause
The default bounding box was set to the **bottom 40% of the image**, starting at **60% down from the top**.

For a 2014×858 image:
- Old default: `y1 = 60% × 858 = 514px`
- This is almost 2/3 down the image - way below the mouth!

## The Fix

### Changed Files:
1. **pipeline.py** (line 338-341)
2. **inference_controlnet.py** (line 576)

### Old (Broken):
```python
# Default: bottom 40% of image
x1, y1 = 0, int(orig_h * 0.6)  # Start at 60% down
x2, y2 = orig_w, orig_h         # End at bottom
```

### New (Fixed):
```python
# Default: mouth region (approximately 45-75% from top)
x1, y1 = 0, int(orig_h * 0.45)  # Start at 45% down
x2, y2 = orig_w, int(orig_h * 0.75)  # End at 75% down
```

## Why 45-75%?

In typical portrait/headshot photos:
- **Top 0-40%**: Forehead, eyes, upper face
- **Middle 40-60%**: Nose, upper lip, mouth area ✓
- **Bottom 60-100%**: Lower lip, chin, neck

Starting at 45% and ending at 75% gives us a **30% vertical slice** centered around the mouth region.

## Verification

For 2014×858 image:
```
OLD:
  y1 = 514px (60% down) ← TOO LOW
  y2 = 858px (100% down)
  Height = 344px

NEW:
  y1 = 386px (45% down) ← Correct for mouth
  y2 = 643px (75% down)
  Height = 257px (30% of image)
```

## When This Matters

This fix applies when:
- ✅ **No bounding box** is sent from frontend (uses default)
- ✅ **User doesn't manually select mouth region**
- ✅ **Automated processing** without face detection

This does NOT affect:
- ❌ Custom bounding boxes sent from frontend (those are used as-is)
- ❌ MediaPipe face landmark detection (overrides default)

## Testing

To test if this is working:

1. **Without bounding box:**
```bash
curl -X POST http://localhost:5001/api/veneer-preview \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_image>",
    "intensity": 0.8,
    "preserve_geometry": true
  }'
```

Check debug output:
```
/Users/joshuawu/VeneerLoungeApp/debug_outputs/run_*/01_original.png
```

The cropped region should show the **mouth area**, not the chin.

2. **With bounding box:**
```bash
curl -X POST http://localhost:5001/api/veneer-preview \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_image>",
    "intensity": 0.8,
    "preserve_geometry": true,
    "bounding_box": {
      "x": 0,
      "y": 50,
      "width": 100,
      "height": 25
    }
  }'
```

This should use the custom bbox (50-75% from top), not the default.

## Image Aspect Ratio Considerations

### Portrait Images (9:16, typical selfie):
- Default 45-75% works well
- Mouth usually at 50-65%

### Landscape Images (16:9, wide):
- Default 45-75% still works
- Mouth usually at 40-60% in landscape headshots

### Square Images (1:1):
- Default 45-75% works well
- Mouth usually centered 45-60%

### Ultra-wide Images (21:9):
- May need frontend to send custom bbox
- Default still reasonable as fallback

## Future Improvements

1. **Face Detection Integration:**
   - Use MediaPipe to detect mouth landmarks
   - Auto-calculate bounding box from landmarks
   - Fall back to default only if detection fails

2. **Adaptive Defaults:**
   - Analyze image aspect ratio
   - Adjust default bbox accordingly
   - Portrait: 45-70%
   - Landscape: 40-65%
   - Square: 45-75%

3. **Frontend Guidance:**
   - Always send bounding box from frontend
   - Use face detection in browser
   - Don't rely on backend default

## Status

✅ **FIXED** - Default bounding box now targets mouth region (45-75%) instead of chin/neck (60-100%).

**Fixed Date:** 2026-02-18
**Priority:** CRITICAL (production-blocking)
**Files Modified:**
- `veneers/pipeline.py` (lines 338-341)
- `controlnet/inference_controlnet.py` (line 576)
