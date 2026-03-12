"""
Veneer Generation Parameter Presets

Pure configuration - no logic.
Each preset defines all parameters for the complete pipeline.

Tuning rationale (calibrated from old working reference):
- guidance_scale 5.5-8.0: moderate guidance avoids gloss blowout
- controlnet_conditioning_scale 0.60-0.75: high ControlNet keeps tooth structure
- strength 0.50-0.75: lower strength preserves more original context
- num_inference_steps 12-30: fewer steps, high controlnet = efficient convergence
- whitening_target 212-220 (LAB L): neutral porcelain white without bloom
- whitening_strength 0.30-0.60: proportional boost, never clips

Trade-offs:
- Higher guidance_scale = sharper teeth but risk of gloss/overexposure
- Lower controlnet_conditioning_scale = more anatomy preservation but less arch correction
- Higher strength = more dramatic transformation but risk of losing lip/gum context
- Higher whitening_target = brighter teeth but risk of blue tint and bloom
"""

PRESETS = {
    "natural": {
        # Diffusion parameters
        # PRODUCTION-GRADE: Lower controlnet + higher strength for geometry freedom
        # Allows actual reconstruction while maintaining facial structure
        "num_inference_steps": 35,
        "guidance_scale": 8.5,
        "controlnet_conditioning_scale": 0.40,  # ← REDUCED from 0.58 for geometry freedom
        "strength": 0.88,  # ← INCREASED from 0.85 for actual change

        # Post-processing parameters
        "whitening_target": 215.0,
        "whitening_strength": 0.45,

        # Edge detection parameters
        "edge_threshold_low": 25,
        "edge_threshold_high": 80,

        # Blending parameters
        "use_poisson_blend": True,
        "feather_inner_px": 8,
        "feather_outer_px": 30,

        # Mask refinement
        "mask_erosion_px": 3,

        # Texture synthesis
        "enamel_texture_strength": 0.32,
    },

    "dramatic": {
        # Diffusion parameters
        # PRODUCTION-GRADE: Minimal controlnet for maximum geometry change
        # Maximum transformation with proper guidance balance
        "num_inference_steps": 40,
        "guidance_scale": 9.5,
        "controlnet_conditioning_scale": 0.35,  # ← REDUCED from 0.48 for freedom
        "strength": 0.92,  # ← INCREASED from 0.90 for major reshaping

        # Post-processing parameters
        "whitening_target": 218.0,
        "whitening_strength": 0.50,

        # Edge detection parameters
        "edge_threshold_low": 20,
        "edge_threshold_high": 70,

        # Blending parameters
        "use_poisson_blend": True,
        "feather_inner_px": 8,
        "feather_outer_px": 30,

        # Mask refinement
        "mask_erosion_px": 3,

        # Texture synthesis
        "enamel_texture_strength": 0.36,
    },

    "conservative": {
        # Diffusion parameters
        # REBALANCED: Medium controlnet + strong denoising for controlled transformation
        # Preserves natural tooth structure while allowing visible whitening/smoothing
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "controlnet_conditioning_scale": 0.55,
        "strength": 0.82,

        # Post-processing parameters
        "whitening_target": 212.0,
        "whitening_strength": 0.40,

        # Edge detection parameters
        "edge_threshold_low": 25,
        "edge_threshold_high": 90,

        # Blending parameters
        "use_poisson_blend": True,
        "feather_inner_px": 8,
        "feather_outer_px": 30,

        # Mask refinement
        "mask_erosion_px": 3,

        # Texture synthesis
        "enamel_texture_strength": 0.30,
    },

    "balanced": {
        # Diffusion parameters
        # PRODUCTION-GRADE: Balanced preset between natural and dramatic
        "num_inference_steps": 30,
        "guidance_scale": 8.0,
        "controlnet_conditioning_scale": 0.42,  # ← Between natural (0.40) and dramatic (0.35)
        "strength": 0.86,  # ← Between natural (0.88) and dramatic (0.92)

        # Post-processing parameters
        "whitening_target": 216.0,
        "whitening_strength": 0.47,

        # Edge detection parameters
        "edge_threshold_low": 25,
        "edge_threshold_high": 80,

        # Blending parameters
        "use_poisson_blend": True,
        "feather_inner_px": 8,
        "feather_outer_px": 30,

        # Mask refinement
        "mask_erosion_px": 3,

        # Texture synthesis
        "enamel_texture_strength": 0.34,
    },

    "fast": {
        # Diffusion parameters
        # Quick preview with good quality
        "num_inference_steps": 12,
        "guidance_scale": 5.5,
        "controlnet_conditioning_scale": 0.75,
        "strength": 0.50,

        # Post-processing parameters
        "whitening_target": 214.0,
        "whitening_strength": 0.42,

        # Edge detection parameters
        "edge_threshold_low": 25,
        "edge_threshold_high": 80,

        # Blending parameters
        "use_poisson_blend": True,
        "feather_inner_px": 8,
        "feather_outer_px": 30,

        # Mask refinement
        "mask_erosion_px": 3,

        # Texture synthesis
        "enamel_texture_strength": 0.28,
    },

    "veneers_max": {
        # Diffusion parameters
        # ULTRA AGGRESSIVE: For testing maximum transformation
        # Very low controlnet + very high strength for extreme geometry change
        "num_inference_steps": 45,
        "guidance_scale": 9.5,
        "controlnet_conditioning_scale": 0.30,  # ← Minimal structure preservation
        "strength": 0.95,  # ← INCREASED from 0.92 for extreme change

        # Post-processing parameters
        "whitening_target": 220.0,
        "whitening_strength": 0.55,

        # Edge detection parameters
        "edge_threshold_low": 20,
        "edge_threshold_high": 70,

        # Blending parameters
        "use_poisson_blend": True,
        "feather_inner_px": 8,
        "feather_outer_px": 30,

        # Mask refinement
        "mask_erosion_px": 3,

        # Texture synthesis
        "enamel_texture_strength": 0.38,
    },
}


# Diagnostic preset for testing - MAXIMUM transformation
PRESETS["test_extreme"] = {
    "num_inference_steps": 40,
    "guidance_scale": 9.0,
    "controlnet_conditioning_scale": 0.25,  # Minimal structure preservation
    "strength": 0.95,  # Maximum denoising
    "whitening_target": 218.0,
    "whitening_strength": 0.50,
    "edge_threshold_low": 30,
    "edge_threshold_high": 90,
    "use_poisson_blend": True,
    "feather_inner_px": 8,
    "feather_outer_px": 30,
    "mask_erosion_px": 2,
    "enamel_texture_strength": 0.35,
}


def get_preset(preset_name="natural"):
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of preset ('natural', 'dramatic', 'conservative', 'fast', 'veneers_max')

    Returns:
        dict: Preset parameters
    """
    return PRESETS.get(preset_name.lower(), PRESETS["natural"]).copy()


def interpolate_presets(preset_a_name, preset_b_name, t):
    """
    Interpolate between two presets.

    Args:
        preset_a_name: Starting preset name
        preset_b_name: Ending preset name
        t: Interpolation factor (0.0 = preset_a, 1.0 = preset_b)

    Returns:
        dict: Interpolated parameters
    """
    preset_a = get_preset(preset_a_name)
    preset_b = get_preset(preset_b_name)

    result = {}
    for key in preset_a:
        if isinstance(preset_a[key], (int, float)):
            result[key] = preset_a[key] + (preset_b[key] - preset_a[key]) * t
        else:
            # Non-numeric values use preset_b if t > 0.5, else preset_a
            result[key] = preset_b[key] if t > 0.5 else preset_a[key]

    return result