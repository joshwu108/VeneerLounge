# Veneer Generation - Modular Architecture

Clean separation of concerns for deterministic veneer generation.

## Architecture Overview

```
veneers/
│
├── engine/                      # Independent engine modules
│   ├── diffusion_engine.py      # SD + ControlNet inference ONLY
│   ├── mask_engine.py           # Segmentation + refinement ONLY
│   ├── conditioning_engine.py   # Ideal arch + spatial conditioning ONLY
│   └── postprocess_engine.py    # Whitening + texture + blending ONLY
│
├── pipeline.py                  # Clean orchestrator (no logic duplication)
├── presets.py                   # Pure configuration (no logic)
├── prompt_templates.py          # Prompt engineering (independent)
└── cli.py                       # Command-line interface
```

## Key Principles

### 1. Single Responsibility
Each engine handles ONE job:
- **DiffusionEngine**: Diffusion inference
- **MaskEngine**: Segmentation
- **ConditioningEngine**: Spatial conditioning (THE CORE INNOVATION)
- **PostProcessor**: Enhancement and blending

### 2. No State Bleeding
- Each engine is independent
- No shared state between modules
- Clean interfaces with explicit parameters

### 3. No Logic Duplication
- Configuration lives in `presets.py`
- Prompts live in `prompt_templates.py`
- Pipeline orchestrates without implementing logic

## The Core Innovation

**ConditioningEngine.generate_ideal_arch()** is the power:
- Generates synthetic perfect dental arch geometry
- Forces diffusion toward ideal veneer alignment
- Overrides original crooked tooth structure

This is what makes the system work. Everything else is plumbing.

## Usage

### As a Library

```python
from veneers import VeneerPipeline

# Initialize pipeline
pipeline = VeneerPipeline(
    controlnet_path="./weights/controlnet",
    base_model="runwayml/stable-diffusion-v1-5",
    segmentation_checkpoint="./weights/seg.pth",
    device="cuda"
)

# Generate veneer preview
result = pipeline.run(
    image="input.jpg",
    preset="natural",  # or 'dramatic', 'conservative', 'fast', 'veneers_max'
    seed=42
)

result.save("output.jpg")
```

### Command Line

```bash
# Basic usage
python -m veneers.cli \
  --controlnet ./weights/controlnet \
  --image input.jpg \
  --output output.jpg

# Dramatic transformation
python -m veneers.cli \
  --controlnet ./weights/controlnet \
  --image input.jpg \
  --output output.jpg \
  --preset dramatic

# With custom prompt
python -m veneers.cli \
  --controlnet ./weights/controlnet \
  --image input.jpg \
  --output output.jpg \
  --preset veneers_max \
  --prompt "perfect Hollywood smile, brilliant white veneers"

# Side-by-side comparison
python -m veneers.cli \
  --controlnet ./weights/controlnet \
  --image input.jpg \
  --output comparison.jpg \
  --comparison
```

## Presets

Available presets in `presets.py`:

- **natural**: Realistic, high-quality veneers (default)
- **dramatic**: Maximum whitening, perfect alignment
- **conservative**: Minimal changes, natural appearance
- **fast**: Quick preview with good quality
- **veneers_max**: Ultimate transformation preset

Each preset defines:
- Diffusion parameters (steps, guidance, strength)
- Post-processing (whitening target/strength)
- Edge detection thresholds
- Texture synthesis strength
- Blending method

## Extending the System

### Add a New Preset

Edit `presets.py`:

```python
PRESETS = {
    "my_preset": {
        "num_inference_steps": 30,
        "guidance_scale": 9.5,
        "controlnet_conditioning_scale": 0.70,
        "strength": 0.90,
        "whitening_target": 217.0,
        "whitening_strength": 0.50,
        "edge_threshold_low": 25,
        "edge_threshold_high": 80,
        "use_poisson_blend": True,
        "feather_inner_px": 8,
        "feather_outer_px": 30,
        "mask_erosion_px": 4,
        "enamel_texture_strength": 0.35,
    }
}
```

### Add a New Prompt Template

Edit `prompt_templates.py`:

```python
PROMPT_TEMPLATES = {
    "my_style": {
        "prompt": "your custom prompt here",
        "negative_prompt": "your negative prompt here"
    }
}
```

### Swap Diffusion Model

Easy to upgrade to SDXL:

```python
from veneers.engine import DiffusionEngine

# In DiffusionEngine.__init__:
# Change base_model to "stabilityai/stable-diffusion-xl-base-1.0"
# Update pipeline to StableDiffusionXLControlNetInpaintPipeline
```

## Benefits Over Monolithic Design

### Before (1200+ lines, monolithic)
- ❌ Impossible to reason about geometry vs diffusion dominance
- ❌ Parameters scattered throughout code
- ❌ Can't test components independently
- ❌ Hard to upgrade or swap models
- ❌ Logic duplication everywhere

### After (modular, ~400 lines total)
- ✅ Clear separation: geometry, diffusion, post-processing
- ✅ All config in one place (`presets.py`)
- ✅ Unit testable engines
- ✅ Easy model upgrades (SDXL, etc.)
- ✅ Zero logic duplication
- ✅ Deterministic and debuggable

## Testing Individual Engines

Each engine can be tested independently:

```python
from veneers.engine import MaskEngine, ConditioningEngine
from PIL import Image

# Test mask generation
mask_engine = MaskEngine(segmentation_checkpoint="weights/seg.pth")
image = Image.open("test.jpg")
mask = mask_engine.generate(image)
mask.save("test_mask.png")

# Test conditioning generation
conditioning_engine = ConditioningEngine()
conditioning = conditioning_engine.build(image, mask)
conditioning.save("test_conditioning.png")
```

## Debug Mode

Enable debug output to see all intermediate steps:

```python
result = pipeline.run(
    image="input.jpg",
    preset="natural",
    debug_dir="./debug_outputs"
)

# Creates:
# debug_outputs/input_crop.png
# debug_outputs/tooth_mask.png
# debug_outputs/conditioning.png
# debug_outputs/output.png
```

## Parameter Tuning Guide

To adjust geometry dominance vs diffusion dominance:

1. **More geometry control** (preserve original structure):
   - ↑ `controlnet_conditioning_scale` (0.7-0.9)
   - ↓ `strength` (0.7-0.85)

2. **More diffusion freedom** (dramatic changes):
   - ↓ `controlnet_conditioning_scale` (0.5-0.65)
   - ↑ `strength` (0.88-0.97)

3. **Whitening intensity**:
   - Adjust `whitening_target` (200-220)
   - Adjust `whitening_strength` (0.2-0.6)

4. **Edge detection sensitivity**:
   - Lower thresholds = more edges detected
   - Higher thresholds = fewer, stronger edges

## Future Enhancements

Easy to add with this architecture:

1. **Automatic occlusion detection** → preset escalation
2. **Dynamic controlnet scale** based on mask asymmetry
3. **Incisal plane flattening enforcement**
4. **GAN-based tooth realism enhancement**
5. **Multi-stage refinement pipeline**

All without touching core engine logic.

## Production Deployment

For startup MVP:
```python
pipeline = VeneerPipeline(
    controlnet_path="./weights/controlnet",
    device="cuda"
)
```

For clinical-grade SaaS:
- Add quality checks after each stage
- Implement automatic preset selection based on input analysis
- Add fallback strategies for edge cases
- Implement batch processing
- Add progress callbacks for UI integration

## License

Proprietary - Veneer Lounge App
