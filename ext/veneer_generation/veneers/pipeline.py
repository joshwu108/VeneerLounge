"""
Veneer Pipeline

Clean orchestrator with NO logic duplication.
Coordinates all engines without state bleeding.

Auto-adjustment features:
- Mask area ratio: auto-attenuates controlnet_conditioning_scale when mask > 25% of image
- Failure detection: checks for lip clipping, artificial brightness, color cast
- Parameter auto-correction based on detected failure modes
- Diffusion delta check: warns when diffusion produces no geometry change

Debug visualization:
- Creates debug/run_TIMESTAMP/ folder with numbered intermediate images
- Saves overlays (mask on original, conditioning overlay, diffusion delta)
- Prints diagnostic metrics at every stage

Two-pass mode (optional):
- Pass 1: Geometry reshape (lower controlnet, higher strength)
- Pass 2: Texture refinement (lower strength, higher guidance)
"""

from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from .engine.diffusion_engine import DiffusionEngine
from .engine.mask_engine import MaskEngine
from .engine.conditioning_engine import ConditioningEngine
from .engine.postprocess_engine import PostProcessor
from .presets import get_preset, interpolate_presets
from .prompt_templates import get_prompts


class VeneerPipeline:
    """
    Main orchestrator for veneer generation.
    Coordinates engines without implementing core logic.
    """

    def __init__(
        self,
        controlnet_path,
        base_model="runwayml/stable-diffusion-v1-5",
        segmentation_checkpoint=None,
        device="cuda"
    ):
        """
        Initialize veneer generation pipeline.

        Args:
            controlnet_path: Path to trained ControlNet weights
            base_model: Base Stable Diffusion model path or HF model ID
            segmentation_checkpoint: Path to segmentation model checkpoint
            device: 'cuda' or 'cpu'
        """
        print("Initializing VeneerPipeline...")

        # Initialize all engines
        self.diffusion = DiffusionEngine(controlnet_path, base_model, device)
        self.mask_engine = MaskEngine(segmentation_checkpoint, device)
        self.conditioning = ConditioningEngine()
        self.post = PostProcessor()

        self.device = device

        print("✓ VeneerPipeline initialized successfully")

    def _auto_adjust_params(self, params, mask_binary_np, image_size):
        """
        Auto-adjust diffusion parameters based on mask properties.

        FIXED: Mask coverage warnings should NOT reduce controlnet strength.
        Large mask coverage usually indicates segmentation failure, not "more room for ControlNet".
        When mask is too large, it's likely including lips/gums, which means we need STRONGER
        controlnet to preserve non-tooth regions, not weaker.

        Only adjust if mask is suspiciously large (>60%, definite segmentation failure).

        Args:
            params: dict of pipeline parameters (modified in-place)
            mask_binary_np: numpy array of binary mask
            image_size: (width, height) of the generation image

        Returns:
            dict: Adjusted parameters
        """
        w, h = image_size
        total_pixels = w * h
        mask_pixels = np.sum(mask_binary_np > 127)
        mask_ratio = mask_pixels / total_pixels

        # Only auto-adjust for extreme mask failure (>60% coverage = definite fail)
        if mask_ratio > 0.60:
            # Mask is way too large - likely complete segmentation failure
            # Reduce strength to prevent modifying the entire image
            params["strength"] = min(params["strength"], 0.70)
            print(f"  Auto-adjust: mask ratio {mask_ratio:.2f} > 0.60 (segmentation likely failed), "
                  f"reducing strength to {params['strength']:.2f} for safety")
        elif mask_ratio < 0.05:
            # Mask is too small - teeth not detected properly
            # Increase controlnet slightly to prevent wild changes to surrounding area
            params["controlnet_conditioning_scale"] = min(
                params["controlnet_conditioning_scale"] * 1.15, 0.85
            )
            print(f"  Auto-adjust: mask ratio {mask_ratio:.2f} < 0.05 (teeth may not be detected), "
                  f"increasing controlnet_scale to {params['controlnet_conditioning_scale']:.3f}")

        return params

    def _detect_failures(self, generated_np, original_crop_np, mask_bool):
        """
        Detect common failure modes in the generated output and return
        suggested parameter adjustments.

        Failure modes detected:
        1. Artificial brightness: mean L in mask > 235 -> reduce whitening_target
        2. Blue/cold tint: mean b channel < 120 -> reduce whitening_strength
        3. Lip clipping: significant mask overlap with lip-colored pixels

        Args:
            generated_np: numpy array of generated image (RGB)
            original_crop_np: numpy array of original crop (RGB)
            mask_bool: numpy boolean mask

        Returns:
            dict: Suggested parameter overrides (empty if no failures detected)
        """
        adjustments = {}

        if not np.any(mask_bool):
            return adjustments

        # Check brightness in LAB
        gen_lab = cv2.cvtColor(generated_np, cv2.COLOR_RGB2LAB)
        l_in_mask = gen_lab[..., 0][mask_bool].astype(np.float32)
        b_in_mask = gen_lab[..., 2][mask_bool].astype(np.float32)

        mean_l = l_in_mask.mean()
        mean_b = b_in_mask.mean()

        # 1. Artificial brightness detection
        if mean_l > 235:
            adjustments["whitening_target_override"] = -8.0  # Reduce by 8
            print(f"  Failure detected: artificial brightness (L={mean_l:.1f})")

        # 2. Blue/cold tint detection (b < 120 means blue shift in OpenCV LAB)
        if mean_b < 118:
            adjustments["whitening_strength_override"] = -0.10  # Reduce by 0.10
            print(f"  Failure detected: blue tint (b={mean_b:.1f})")

        # 3. Check if mask extends into lip-colored region
        # Lip pixels tend to have high a-channel (red) in LAB
        orig_lab = cv2.cvtColor(original_crop_np, cv2.COLOR_RGB2LAB)
        a_in_mask = orig_lab[..., 1][mask_bool].astype(np.float32)
        # If >20% of mask pixels have strong red (a > 145), mask may include lips
        lip_ratio = np.mean(a_in_mask > 145)
        if lip_ratio > 0.20:
            adjustments["mask_erosion_extra"] = 2  # Erode mask by 2 more pixels
            print(f"  Failure detected: lip clipping ({lip_ratio:.1%} lip pixels in mask)")

        return adjustments

    def _check_diffusion_delta(self, input_np, output_np, mask_bool):
        """
        Check whether diffusion actually modified the image geometry.

        Computes mean absolute difference between input and output within
        the mask region. If delta is below threshold, prints warnings with
        parameter tuning recommendations.

        Args:
            input_np: numpy array of diffusion input (RGB, uint8)
            output_np: numpy array of diffusion output (RGB, uint8)
            mask_bool: numpy boolean mask (True = teeth region)

        Returns:
            float: Mean absolute difference within mask
        """
        if not np.any(mask_bool):
            return 0.0

        input_f = input_np.astype(np.float32)
        output_f = output_np.astype(np.float32)

        # Compute per-pixel absolute difference
        diff = np.abs(output_f - input_f)

        # Mean difference within mask only (across all 3 channels)
        mask_3d = mask_bool[..., np.newaxis]
        masked_diff = diff[np.broadcast_to(mask_3d, diff.shape)]
        mean_delta = masked_diff.mean()

        print(f"  Diffusion delta (mean abs diff in mask): {mean_delta:.2f}")

        if mean_delta < 3.0:
            print("  WARNING: Diffusion likely constrained by ControlNet or low strength.")
            print("  Recommendations:")
            print("    1. Reduce controlnet_conditioning_scale to 0.35")
            print("    2. If still unchanged, increase strength to 0.90")
            print("    3. Verify mask is not mostly black (check debug images)")
            print("    4. Verify inpaint model is actually inpaint-capable")
        elif mean_delta < 8.0:
            print("  NOTE: Diffusion delta is low — geometry change may be minimal.")
            print("  Consider reducing controlnet_conditioning_scale by 0.05-0.10")

        return mean_delta

    def _save_debug_images(self, debug_path, stage_name, image, number):
        """Save a numbered debug image."""
        filename = f"{number:02d}_{stage_name}.png"
        if isinstance(image, Image.Image):
            image.save(debug_path / filename)
        else:
            Image.fromarray(image).save(debug_path / filename)

    def _create_overlay(self, base_np, mask_np, color=(0, 255, 0), alpha=0.4):
        """Create a colored overlay of mask on the base image."""
        overlay = base_np.copy()
        mask_bool = mask_np > 127 if mask_np.ndim == 2 else mask_np[..., 0] > 127
        colored = np.zeros_like(overlay)
        colored[mask_bool] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
        return overlay

    def _create_lip_polygon_overlay(self, crop_resized, lip_landmarks, crop_box,
                                       full_image_size, gen_size):
        """Create debug visualization of lip polygon projected onto crop."""
        overlay = np.array(crop_resized).copy()
        x1, y1, x2, y2 = crop_box
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w <= 0 or crop_h <= 0:
            return overlay
        scale_x = gen_size / crop_w
        scale_y = gen_size / crop_h

        def transform(points):
            return [
                (int((px - x1) * scale_x), int((py - y1) * scale_y))
                for px, py in points
            ]

        inner_upper = transform(lip_landmarks["inner_upper_lip"])
        inner_lower = transform(lip_landmarks["inner_lower_lip"])

        # Draw inner lip polygon (green)
        poly = np.array(inner_upper + inner_lower[::-1], dtype=np.int32)
        cv2.polylines(overlay, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw outer lip landmarks (red dots) for reference
        if "outer_upper_lip" in lip_landmarks:
            outer_upper = transform(lip_landmarks["outer_upper_lip"])
            outer_lower = transform(lip_landmarks["outer_lower_lip"])
            for pt in outer_upper + outer_lower:
                cv2.circle(overlay, pt, 3, (255, 0, 0), -1)

        return overlay

    def _create_delta_image(self, input_np, output_np):
        """
        Create a visualization of the absolute difference between input and output.
        Amplified 3x for visibility.
        """
        diff = np.abs(input_np.astype(np.float32) - output_np.astype(np.float32))
        diff_amplified = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
        return diff_amplified

    def run(
        self,
        image,
        preset="natural",
        prompt_template="default",
        custom_prompt=None,
        custom_negative_prompt=None,
        seed=None,
        debug_dir=None,
        bounding_box=None,
        diagnostic_edges_only=False,
        two_pass=False
    ):
        """
        Run the complete veneer generation pipeline.

        Args:
            image: PIL Image or path to image
            preset: Preset name ('natural', 'dramatic', 'conservative', 'fast', 'veneers_max')
            prompt_template: Prompt template name
            custom_prompt: Custom text prompt override
            custom_negative_prompt: Custom negative prompt override
            seed: Random seed for reproducibility
            debug_dir: Directory to save debug images (creates timestamped subfolder)
            bounding_box: Dict with 'x', 'y', 'width', 'height' (percentages)
            diagnostic_edges_only: If True, uses edges-only conditioning (no ideal arch) for diagnostic
            two_pass: If True, uses 2-pass system (geometry reshape then texture refinement)

        Returns:
            PIL Image: Final veneer preview
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Set up debug directory with timestamp
        debug_path = None
        if debug_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_path = Path(debug_dir) / f"run_{timestamp}"
            debug_path.mkdir(parents=True, exist_ok=True)
            print(f"  Debug output: {debug_path}")

        # Get preset parameters
        params = get_preset(preset)
        print(f"  Preset: {preset}")
        print(f"  Parameters: steps={params['num_inference_steps']}, "
              f"guidance={params['guidance_scale']}, "
              f"controlnet={params['controlnet_conditioning_scale']}, "
              f"strength={params['strength']}")

        # Get prompts
        prompt, negative_prompt = get_prompts(
            prompt_template,
            custom_prompt,
            custom_negative_prompt
        )

        # Process bounding box
        orig_w, orig_h = image.size
        if bounding_box:
            x = int(bounding_box["x"] / 100.0 * orig_w)
            y = int(bounding_box["y"] / 100.0 * orig_h)
            w = int(bounding_box["width"] / 100.0 * orig_w)
            h = int(bounding_box["height"] / 100.0 * orig_h)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(orig_w, x + w), min(orig_h, y + h)
        else:
            # Default: mouth region (approximately 45-75% from top)
            # This targets the lower-middle region where the mouth typically appears
            x1, y1 = 0, int(orig_h * 0.45)
            x2, y2 = orig_w, int(orig_h * 0.75)

        # Detect face landmarks from full image (before cropping)
        # MediaPipe needs the full face for reliable lip detection
        lip_landmarks = self.mask_engine.detect_face_landmarks(image)
        if lip_landmarks:
            print(f"  Face landmarks detected — using MediaPipe lip-guided masking")
        else:
            print(f"  No face landmarks — using fallback segmentation")

        crop_box = (x1, y1, x2, y2)
        full_image_size = (orig_w, orig_h)

        # Crop to bounding box
        crop = image.crop((x1, y1, x2, y2))

        GEN_SIZE = 512
        crop_resized = crop.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)

        # --- Debug: 01_original ---
        if debug_path:
            self._save_debug_images(debug_path, "original", crop_resized, 1)

        # Step 1: Get raw segmentation (before refinement) for debug
        if debug_path:
            raw_mask = self.mask_engine.segment_raw(
                crop_resized, target_size=(GEN_SIZE, GEN_SIZE),
                lip_landmarks=lip_landmarks,
                crop_box=crop_box,
                full_image_size=full_image_size
            )
            self._save_debug_images(debug_path, "raw_mask", raw_mask, 2)

        # Step 1: Generate mask (binary, not feathered yet)
        mask = self.mask_engine.generate(
            crop_resized,
            target_size=(GEN_SIZE, GEN_SIZE),
            erosion_px=params["mask_erosion_px"],
            lip_landmarks=lip_landmarks,
            crop_box=crop_box,
            full_image_size=full_image_size
        )

        # Check if mask is valid, use fallback if needed
        mask_np = np.array(mask)
        if mask_np.mean() < 10:
            print("  WARNING: Mask is nearly empty — using fallback center rectangle")
            mask = self.mask_engine.create_fallback_mask(size=(GEN_SIZE, GEN_SIZE))
            mask_np = np.array(mask)

        # --- Debug: 03_binary_mask (refined) ---
        if debug_path:
            self._save_debug_images(debug_path, "binary_mask", mask, 3)

        # Auto-adjust parameters based on mask area ratio
        params = self._auto_adjust_params(params, mask_np, (GEN_SIZE, GEN_SIZE))

        # Step 1b: Feather the mask for smooth blending
        mask_feathered = self.mask_engine.feather(
            mask,
            inner_feather_px=params.get("feather_inner_px", 8),
            outer_feather_px=params.get("feather_outer_px", 30)
        )

        # --- Debug: 04_eroded_mask (same as binary since erosion is in refinement) ---
        # --- Debug: 05_distance_mask (feathered) ---
        if debug_path:
            self._save_debug_images(debug_path, "eroded_mask", mask, 4)
            self._save_debug_images(debug_path, "distance_mask", mask_feathered, 5)

        # Step 2: Generate conditioning (use binary mask, not feathered)
        conditioning = self.conditioning.build(
            crop_resized,
            mask,  # Binary mask for conditioning
            edge_threshold_low=params["edge_threshold_low"],
            edge_threshold_high=params["edge_threshold_high"],
            use_ideal_arch=not diagnostic_edges_only
        )

        # --- Debug: 06_controlnet_condition ---
        if debug_path:
            self._save_debug_images(debug_path, "controlnet_condition", conditioning, 6)

        # --- Debug: 07_inpaint_input ---
        if debug_path:
            self._save_debug_images(debug_path, "inpaint_input", crop_resized, 7)

        # --- Two-pass or single-pass diffusion ---
        # Auto-enable two-pass for aggressive presets
        auto_two_pass = preset in ["dramatic", "veneers_max"] or two_pass

        if auto_two_pass:
            print(f"  Using two-pass mode (preset={preset})")
            generated = self._run_two_pass(
                crop_resized, mask_feathered, conditioning,
                prompt, negative_prompt, params, seed, debug_path
            )
        else:
            # Step 3: Run diffusion (use feathered mask for inpainting)
            generated = self.diffusion.generate(
                image=crop_resized,
                mask=mask_feathered,
                conditioning=conditioning,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                controlnet_conditioning_scale=params["controlnet_conditioning_scale"],
                strength=params["strength"],
                seed=seed
            )

        # Sanitize NaN/Inf values from diffusion output (MPS float16 can produce these)
        gen_np = np.array(generated).astype(np.float32)
        if np.any(~np.isfinite(gen_np)):
            print("  Warning: NaN/Inf detected in diffusion output — replacing with original pixels")
            orig_np = np.array(crop_resized).astype(np.float32)
            gen_np = np.where(np.isfinite(gen_np), gen_np, orig_np)
            generated = Image.fromarray(np.clip(gen_np, 0, 255).astype(np.uint8))

        # --- Debug: 08_raw_diffusion_output ---
        if debug_path:
            self._save_debug_images(debug_path, "raw_diffusion_output", generated, 8)

        # --- Diffusion delta check (PART 7) ---
        mask_bool = mask_np > 127
        input_crop_np = np.array(crop_resized)
        output_crop_np = np.array(generated)

        # Compute LAB L channel before/after for diagnostics
        if np.any(mask_bool):
            input_lab = cv2.cvtColor(input_crop_np, cv2.COLOR_RGB2LAB)
            output_lab = cv2.cvtColor(output_crop_np, cv2.COLOR_RGB2LAB)
            input_mean_l = input_lab[..., 0][mask_bool].astype(np.float32).mean()
            output_mean_l = output_lab[..., 0][mask_bool].astype(np.float32).mean()
            print(f"  LAB L channel — input: {input_mean_l:.1f}, output: {output_mean_l:.1f}")

        delta = self._check_diffusion_delta(input_crop_np, output_crop_np, mask_bool)

        # Auto-retry if delta is too low (diffusion ineffective)
        if delta < 5.0 and not auto_two_pass and not hasattr(self, '_retry_attempted'):
            print("  🔄 Auto-retry: Diffusion delta too low — reducing ControlNet to 0.30, increasing strength to 0.92")
            self._retry_attempted = True  # Prevent infinite loop

            # Re-run diffusion with corrected parameters
            generated = self.diffusion.generate(
                image=crop_resized,
                mask=mask_feathered,
                conditioning=conditioning,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=params["num_inference_steps"],
                guidance_scale=params["guidance_scale"],
                controlnet_conditioning_scale=0.30,  # ← Force low ControlNet
                strength=0.92,  # ← Force high strength
                seed=seed
            )

            # Recompute delta
            output_crop_np = np.array(generated)
            delta = self._check_diffusion_delta(input_crop_np, output_crop_np, mask_bool)
            print(f"  Retry complete — new delta: {delta:.2f}")

            # Update debug image if needed
            if debug_path:
                self._save_debug_images(debug_path, "raw_diffusion_output_retry", generated, 8)

        # --- Debug: diffusion delta visualization ---
        if debug_path:
            delta_image = self._create_delta_image(input_crop_np, output_crop_np)
            self._save_debug_images(debug_path, "diffusion_delta", delta_image, 15)

            # Save conditioning channels separately for detailed analysis
            cond_np = np.array(conditioning)
            Image.fromarray(cond_np[..., 0], mode='L').save(debug_path / "16_cond_mask_channel.png")
            Image.fromarray(cond_np[..., 1], mode='L').save(debug_path / "17_cond_edges_channel.png")
            Image.fromarray(cond_np[..., 2], mode='L').save(debug_path / "18_cond_distance_channel.png")
            print(f"  Conditioning channels saved: R=mask, G=edges/arch, B=distance")

        # Step 3b: Failure detection and auto-correction
        failures = self._detect_failures(
            np.array(generated),
            np.array(crop_resized),
            mask_bool
        )

        # Apply failure corrections to whitening parameters
        wt = params["whitening_target"]
        ws = params["whitening_strength"]
        if "whitening_target_override" in failures:
            wt += failures["whitening_target_override"]
        if "whitening_strength_override" in failures:
            ws += failures["whitening_strength_override"]
            ws = max(0.15, ws)  # Floor to prevent zero whitening

        # Step 4: Post-processing enhancement (use binary mask for precise whitening)
        enhanced = self.post.enhance(
            generated,
            mask,
            whitening_target=wt,
            whitening_strength=ws,
            enamel_texture_strength=params["enamel_texture_strength"]
        )

        # --- Debug: 09_post_whiten ---
        if debug_path:
            self._save_debug_images(debug_path, "post_whiten", enhanced, 9)

        # Step 5: Resize back to original crop size
        bbox_w, bbox_h = x2 - x1, y2 - y1
        enhanced_resized = enhanced.resize((bbox_w, bbox_h), Image.LANCZOS)
        mask_feathered_resized = mask_feathered.resize((bbox_w, bbox_h), Image.LANCZOS)

        # Step 6: Blend back into original (use feathered mask for smooth compositing)
        result = self.post.blend(
            original=image,
            generated=enhanced_resized,
            mask=mask_feathered_resized,
            offset=(x1, y1),
            use_poisson=params["use_poisson_blend"]
        )

        # --- Debug: 10_poisson_blend and 11_final_result ---
        if debug_path:
            self._save_debug_images(debug_path, "poisson_blend", result, 10)
            self._save_debug_images(debug_path, "final_result", result, 11)

            # --- Overlay images ---
            # Mask overlay on original
            mask_overlay = self._create_overlay(
                np.array(crop_resized), mask_np, color=(0, 255, 0), alpha=0.4
            )
            self._save_debug_images(debug_path, "overlay_mask_on_original", mask_overlay, 12)

            # Conditioning overlay on mask
            cond_np = np.array(conditioning)
            cond_overlay = self._create_overlay(
                np.array(crop_resized), cond_np[..., 0], color=(255, 0, 0), alpha=0.3
            )
            self._save_debug_images(debug_path, "overlay_conditioning", cond_overlay, 13)

            # Lip polygon overlay (if MediaPipe landmarks available)
            if lip_landmarks:
                poly_overlay = self._create_lip_polygon_overlay(
                    crop_resized, lip_landmarks, crop_box, full_image_size, GEN_SIZE
                )
                self._save_debug_images(debug_path, "lip_polygon_overlay", poly_overlay, 14)

            print(f"  Debug images saved to {debug_path}")

        return result

    def _run_two_pass(
        self, crop_resized, mask_feathered, conditioning,
        prompt, negative_prompt, params, seed, debug_path
    ):
        """
        Two-pass diffusion system for stronger geometry change.

        Pass 1 — Geometry reshape:
            Lower controlnet_conditioning_scale (0.35) gives diffusion more freedom
            to deviate from original tooth structure. Higher strength (0.88) increases
            the denoising range so more of the image can change.

        Pass 2 — Texture refinement:
            Takes Pass 1 output as new input. Lower strength (0.70) preserves the
            geometry from Pass 1 while refining surface detail. Higher guidance (10.5)
            pushes harder on prompt adherence for porcelain texture quality.

        Args:
            crop_resized: PIL Image (input crop)
            mask_feathered: PIL Image (feathered mask)
            conditioning: PIL Image (ControlNet conditioning)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            params: Pipeline parameters
            seed: Random seed
            debug_path: Path for debug images

        Returns:
            PIL Image: Final generated output after both passes
        """
        print("  === Two-pass mode ===")

        # Pass 1: Geometry reshape
        # Lower controlnet gives diffusion more freedom to deviate from original tooth structure
        # Higher strength increases the denoising range for more dramatic change
        print("  Pass 1: Geometry reshape (controlnet=0.35, strength=0.90)")
        pass1_output = self.diffusion.generate(
            image=crop_resized,
            mask=mask_feathered,
            conditioning=conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=35,
            guidance_scale=8.5,
            controlnet_conditioning_scale=0.35,
            strength=0.90,
            seed=seed
        )

        if debug_path:
            self._save_debug_images(debug_path, "pass1_geometry", pass1_output, 20)

        # Pass 2: Texture refinement (use Pass 1 output as input)
        # Lower strength preserves Pass 1 geometry while refining surface detail
        # Higher guidance pushes harder on prompt adherence for porcelain texture quality
        print("  Pass 2: Texture refinement (controlnet=0.50, strength=0.65, guidance=9.5)")
        pass2_output = self.diffusion.generate(
            image=pass1_output,
            mask=mask_feathered,
            conditioning=conditioning,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=9.5,
            controlnet_conditioning_scale=0.50,
            strength=0.65,
            seed=seed
        )

        if debug_path:
            self._save_debug_images(debug_path, "pass2_texture", pass2_output, 21)

        return pass2_output

    def generate_comparison(self, image, output_path=None, **kwargs):
        """
        Generate side-by-side comparison of before/after.

        Args:
            image: Input image
            output_path: Path to save comparison (optional)
            **kwargs: Arguments for run()

        Returns:
            PIL Image: Comparison image
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Resize to standard size for comparison
        image_resized = image.resize((512, 512), Image.LANCZOS)
        preview = self.run(image_resized, **kwargs)

        # Create side-by-side comparison
        comparison = Image.new("RGB", (1024, 512))
        comparison.paste(image_resized, (0, 0))
        comparison.paste(preview, (512, 0))

        if output_path:
            comparison.save(output_path, quality=95)
            print(f"✓ Comparison saved to {output_path}")

        return comparison
