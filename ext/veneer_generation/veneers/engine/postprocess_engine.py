"""
Post-Processing Engine

Handles ONLY post-generation enhancement and blending.
No diffusion, no segmentation, no conditioning.

Enhancement pipeline:
1. Enamel micro-texture (multi-frequency noise with warm bias)
2. LAB whitening with anti-bloom (L clamped at 230, highlight compression)
3. Chromaticity nudge toward neutral (35-40% pull to a=128, b=128)

Blending pipeline:
1. Poisson blending (cv2.seamlessClone) - primary method
2. Alpha blending with histogram matching in LAB - fallback
3. Edge-aware guided/bilateral filter for gumline sharpness
"""

import numpy as np
import cv2
from PIL import Image


class PostProcessor:
    """
    Manages post-generation enhancement and blending.
    Single responsibility: whitening, texture synthesis, and compositing.
    """

    def __init__(self):
        """Initialize post-processor."""
        pass

    def enhance(self, generated, mask, whitening_target=215.0, whitening_strength=0.45, enamel_texture_strength=0.30):
        """
        Enhance generated image with enamel texture and whitening.

        Args:
            generated: PIL Image (generated teeth)
            mask: PIL Image (tooth mask, binary preferred)
            whitening_target: Target L value in LAB space (212-220 recommended)
            whitening_strength: Whitening boost factor (0.30-0.60 recommended)
            enamel_texture_strength: Enamel texture intensity

        Returns:
            PIL Image: Enhanced result
        """
        gen_np = np.array(generated)
        mask_np = np.array(mask)
        mask_bool = mask_np > 127

        # Add enamel micro-texture
        gen_np = self._add_enamel_texture(gen_np, mask_bool, enamel_texture_strength)

        # Apply LAB whitening
        gen_np = self._whiten_lab(gen_np, mask_bool, whitening_target, whitening_strength)

        return Image.fromarray(gen_np)

    def blend(self, original, generated, mask, offset=(0, 0), use_poisson=True):
        """
        Blend generated teeth back into original image.

        Args:
            original: PIL Image (full original image)
            generated: PIL Image (generated teeth crop)
            mask: PIL Image (tooth mask, same size as generated)
            offset: (x, y) offset for pasting
            use_poisson: Use Poisson blending if True, else alpha blending

        Returns:
            PIL Image: Final composited result
        """
        x0, y0 = offset
        orig_np = np.array(original)
        gen_np = np.array(generated)
        mask_np = np.array(mask)

        # Find the tight bounding box of the actual tooth mask.
        # CRITICAL: only paste within this tight region, NOT the full bounding box.
        # SD modifies the entire crop (including chin/jaw outside the teeth mask),
        # and pasting the full bbox back creates a visible rectangular seam.
        tooth_rows = np.where(mask_np.max(axis=1) > 10)[0]
        tooth_cols = np.where(mask_np.max(axis=0) > 10)[0]

        h_full, w_full = gen_np.shape[:2]

        if len(tooth_rows) > 0 and len(tooth_cols) > 0:
            pad = 35  # pixels of padding so feathering has room to fade
            tr0 = max(0, int(tooth_rows.min()) - pad)
            tr1 = min(h_full, int(tooth_rows.max()) + pad)
            tc0 = max(0, int(tooth_cols.min()) - pad)
            tc1 = min(w_full, int(tooth_cols.max()) + pad)
        else:
            tr0, tr1, tc0, tc1 = 0, h_full, 0, w_full

        # Slice all arrays to the tight tooth region
        gen_tight  = gen_np[tr0:tr1, tc0:tc1]
        mask_tight = mask_np[tr0:tr1, tc0:tc1]
        orig_tight = orig_np[y0+tr0:y0+tr1, x0+tc0:x0+tc1]

        if use_poisson:
            result_tight = self._poisson_blend(gen_tight, orig_tight, mask_tight)
            if result_tight is not None:
                result_np = np.array(original.copy())
                result_np[y0+tr0:y0+tr1, x0+tc0:x0+tc1] = result_tight
                return Image.fromarray(result_np)

        # Fallback: alpha blending on the tight region only
        result_tight = self._alpha_blend(gen_tight, orig_tight, mask_tight)
        result_np = np.array(original.copy())
        result_np[y0+tr0:y0+tr1, x0+tc0:x0+tc1] = result_tight
        return Image.fromarray(result_np)

    def _add_enamel_texture(self, image_np, mask_bool, strength=0.30):
        """
        Add realistic enamel micro-texture using multi-frequency noise.

        Three texture layers:
        1. Fine grain (Perlin-like upsampled noise) - general tooth surface
        2. Vertical streaks - enamel micro-ridges (perikymata)
        3. Distance-based falloff - more translucency/texture at incisal edges

        Warm bias: R=1.0, G=0.95, B=0.9 to avoid cold/blue appearance.

        Args:
            image_np: numpy array (RGB image)
            mask_bool: numpy array (boolean mask)
            strength: Texture intensity (0.25-0.40 recommended)

        Returns:
            numpy array: Image with texture
        """
        h, w = mask_bool.shape

        if not np.any(mask_bool):
            return image_np

        # 1. Fine grain: Perlin-like via upsampled small noise
        small_h, small_w = max(1, h // 16), max(1, w // 16)
        small = np.random.randn(small_h, small_w).astype(np.float32)
        fine_texture = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        fine_texture = cv2.GaussianBlur(fine_texture, (5, 5), 1.5)

        # 2. Vertical streaks: enamel has vertical micro-ridges
        streaks = np.random.randn(1, w).astype(np.float32)
        streaks = np.tile(streaks, (h, 1))
        streaks = cv2.GaussianBlur(streaks, (3, 15), 0)  # Tall, narrow kernel

        # 3. Combine with distance-based falloff (more texture at edges)
        dist = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
        dist_max = dist.max()
        dist_norm = np.clip(dist / dist_max if dist_max > 0 else dist, 0, 1)
        # More texture near edges (translucency), less at center
        edge_weight = 1.0 - dist_norm * 0.6

        texture = (fine_texture * 0.6 + streaks * 0.4) * edge_weight * strength
        # Warm bias: slightly less texture in G and B for natural enamel warmth
        texture_3d = texture[..., np.newaxis] * np.array([1.0, 0.95, 0.9])

        result = image_np.astype(np.float32)
        # Apply texture only inside mask via broadcasting
        mask_f = mask_bool.astype(np.float32)[..., np.newaxis]
        result += texture_3d * 3.0 * mask_f

        return np.clip(result, 0, 255).astype(np.uint8)

    def _whiten_lab(self, image_np, mask_bool, target_l=215.0, strength=0.45):
        """
        Clamped LAB whitening with anti-bloom safeguards.

        Algorithm:
        1. Convert to LAB, extract L/a/b channels
        2. Compute current mean L in mask region
        3. Proportional boost: (target - current_mean) * strength
        4. Clamp L to [current_mean * 0.9, 230] — hard ceiling prevents bloom
        5. Anti-bloom: pixels above L=230 get highlight compression (excess * 0.3)
        6. Chromaticity nudge: pull a/b channels 35% toward neutral (128)
           - This removes blue/yellow tint while preserving some natural warmth

        Args:
            image_np: numpy array (RGB image)
            mask_bool: numpy array (boolean mask)
            target_l: Target L value in LAB space (212-220 recommended)
            strength: Whitening strength factor (0.30-0.60 recommended)

        Returns:
            numpy array: Whitened image
        """
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[..., 0].astype(np.float32)
        a_channel = lab[..., 1].astype(np.float32)
        b_channel = lab[..., 2].astype(np.float32)

        if np.any(mask_bool):
            current_mean = l_channel[mask_bool].mean()

            # Proportional boost capped at L=230 to prevent bloom
            boost = (target_l - current_mean) * strength
            l_channel[mask_bool] = np.clip(
                l_channel[mask_bool] + boost,
                current_mean * 0.9,  # Don't darken below 90% of current mean
                230.0                 # Hard ceiling prevents blown highlights
            )

            # Nudge chromaticity toward neutral (a=128, b=128 in OpenCV LAB 0-255 scale)
            # Pull 35% toward neutral, preserving 65% of original tone
            # (Reduced from 40% to avoid washing out natural warmth)
            a_neutral = 128.0
            b_neutral = 128.0
            a_channel[mask_bool] = a_channel[mask_bool] * 0.65 + a_neutral * 0.35
            b_channel[mask_bool] = b_channel[mask_bool] * 0.65 + b_neutral * 0.35

            # Anti-bloom: soft-clamp highlights above L=228
            # Compress excess brightness to prevent hot spots
            hot_pixels = mask_bool & (l_channel > 228)
            if np.any(hot_pixels):
                excess = l_channel[hot_pixels] - 228
                l_channel[hot_pixels] = 228 + excess * 0.25  # Aggressive highlight compression

        lab[..., 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
        lab[..., 1] = np.clip(a_channel, 0, 255).astype(np.uint8)
        lab[..., 2] = np.clip(b_channel, 0, 255).astype(np.uint8)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _poisson_blend(self, generated_crop, original_crop, mask, use_mixed=False):
        """
        Seamless cloning using Poisson editing for gradient-domain compositing.

        Poisson blending solves the Laplace equation with boundary conditions
        from the original image, producing seamless transitions without haloing.

        Args:
            generated_crop: numpy array (generated teeth region)
            original_crop: numpy array (original image crop)
            mask: numpy array (grayscale mask)
            use_mixed: Use MIXED_CLONE for better texture preservation at boundaries

        Returns:
            numpy array: Blended image, or None if failed
        """
        # Ensure mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Erode mask slightly to avoid edge artifacts from Poisson solver
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

        # Verify mask has enough pixels for Poisson solve
        if binary_mask.sum() < 100 * 255:
            return None

        # Calculate center of mass of the mask for accurate placement
        h, w = binary_mask.shape
        moments = cv2.moments(binary_mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            center = (cx, cy)
        else:
            center = (w // 2, h // 2)

        try:
            clone_mode = cv2.MIXED_CLONE if use_mixed else cv2.NORMAL_CLONE
            result = cv2.seamlessClone(
                generated_crop,
                original_crop,
                binary_mask,
                center,
                clone_mode
            )
            return result
        except Exception as e:
            print(f"Poisson blending failed: {e}, falling back to alpha blending")
            return None

    def _alpha_blend(self, generated_crop, original_crop, mask):
        """
        Advanced alpha blending with histogram matching in LAB transition zones.

        Pipeline:
        1. Distance-based feathering (20px zone) for smooth falloff
        2. Edge-aware guided/bilateral filter for gumline sharpness
        3. Histogram matching in LAB at outer boundary (alpha 0.02-0.3)
        4. Color correction with spatial falloff for seamless transition
        5. Linear alpha compositing

        Args:
            generated_crop: numpy array (generated teeth region)
            original_crop: numpy array (original image crop)
            mask: numpy array (grayscale mask)

        Returns:
            numpy array: Blended image
        """
        mask_np = mask.astype(np.float32) / 255.0

        # Distance-based feathering with narrower zone (less halo)
        mask_binary = (mask_np > 0.05).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)

        feather_width = 20  # Narrower to reduce halo effect
        smooth_mask = np.clip(dist_transform / feather_width, 0, 1)

        # Edge-aware filter preserves gumline sharpness
        smooth_mask_8u = (smooth_mask * 255).astype(np.uint8)
        try:
            # Try guided filter if opencv-contrib is available
            orig_gray = cv2.cvtColor(original_crop, cv2.COLOR_RGB2GRAY)
            smooth_mask_8u = cv2.ximgproc.guidedFilter(
                guide=orig_gray, src=smooth_mask_8u, radius=8, eps=100
            )
        except AttributeError:
            # Fall back to bilateral filter (standard opencv)
            smooth_mask_8u = cv2.bilateralFilter(smooth_mask_8u, d=15, sigmaColor=30, sigmaSpace=15)
        smooth_mask = smooth_mask_8u.astype(np.float32) / 255.0

        orig_crop_f = original_crop.astype(np.float32)
        gen_crop_f = generated_crop.astype(np.float32)

        # Color and brightness matching in transition zones
        alpha_3d = smooth_mask[..., np.newaxis]

        # Define transition zone (where blending happens)
        transition_zone = (alpha_3d[..., 0] > 0.02) & (alpha_3d[..., 0] < 0.98)

        if np.any(transition_zone):
            # Match histogram in LAB space for perceptually correct blending
            gen_lab = cv2.cvtColor(generated_crop.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            orig_lab = cv2.cvtColor(original_crop.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

            # Get boundary pixels (outer transition zone)
            outer_boundary = (alpha_3d[..., 0] > 0.02) & (alpha_3d[..., 0] < 0.3)

            if np.any(outer_boundary):
                for c in range(3):
                    gen_boundary = gen_lab[..., c][outer_boundary]
                    orig_boundary = orig_lab[..., c][outer_boundary]

                    if len(gen_boundary) > 10:
                        gen_mean, gen_std = gen_boundary.mean(), gen_boundary.std() + 1e-6
                        orig_mean, orig_std = orig_boundary.mean(), orig_boundary.std() + 1e-6

                        # Apply color correction with spatial falloff
                        scale = orig_std / gen_std
                        shift = orig_mean - gen_mean * scale

                        # Gradual correction (stronger at edges, weaker at center)
                        correction_strength = 1.0 - alpha_3d[..., 0]
                        gen_lab[..., c] = gen_lab[..., c] * (1.0 + (scale - 1.0) * correction_strength * 0.4) + shift * correction_strength * 0.4

            # Convert back to RGB
            gen_lab = np.clip(gen_lab, 0, 255).astype(np.uint8)
            gen_crop_f = cv2.cvtColor(gen_lab, cv2.COLOR_LAB2RGB).astype(np.float32)

        # Final alpha blending
        blended = (gen_crop_f * alpha_3d + orig_crop_f * (1.0 - alpha_3d))
        return np.clip(blended, 0, 255).astype(np.uint8)