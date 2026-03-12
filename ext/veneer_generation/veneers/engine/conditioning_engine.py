"""
Conditioning Engine

Handles ONLY spatial conditioning generation.
This is the core innovation: synthetic ideal-arch conditioning.
No diffusion, no segmentation, no post-processing.

Design rationale:
- Arch line intensities 100-160: visible guidance without blueprint enforcement
- Gaussian blur (9,9)/sigma 2.5: softens hard pixel edges into smooth gradients
- Separator lines are thin (thickness 1) to suggest, not dictate
- Occlusion plane is subtle (intensity 120) to hint at bite plane
- All geometry masked to teeth region to prevent lip/gum contamination
"""

import numpy as np
import cv2
from PIL import Image


class ConditioningEngine:
    """
    Manages spatial conditioning generation for ControlNet.
    build multi-channel conditioning images.
    """

    def __init__(self):
        """Initialize conditioning engine."""
        pass

    def build(self, image, mask, edge_threshold_low=25, edge_threshold_high=80, use_ideal_arch=True):
        """
        Build multi-channel conditioning image.
        Args:
            image: PIL Image (input photo)
            mask: PIL Image (tooth mask, grayscale)
            edge_threshold_low: Canny low threshold
            edge_threshold_high: Canny high threshold
            use_ideal_arch: If False, uses edges-only for diagnostic (no ideal arch)
        Returns:
            PIL Image: 3-channel conditioning (R=mask, G=ideal_arch+edges, B=distance)
        """
        # Enforce strictly binary mask (0 or 255, no soft gradients)
        mask_np = np.array(mask).astype(np.uint8)
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        mask_bool = mask_binary > 0

        # Real edges from the photograph
        edges = self._generate_edges(image, edge_threshold_low, edge_threshold_high)
        edges_np = edges.astype(np.uint8)

        if use_ideal_arch:
            # Ideal arch: moderate-strength guidance for veneer alignment
            ideal_arch = self._generate_ideal_arch(Image.fromarray(mask_binary))

            if ideal_arch.shape != edges_np.shape:
                ideal_arch = cv2.resize(
                    ideal_arch,
                    (edges_np.shape[1], edges_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            # Inside mask: ideal arch replaces original tooth edges
            # Outside mask: preserve real edges (lips, face structure)
            edges_combined = np.where(mask_bool, ideal_arch, edges_np)
        else:
            # DIAGNOSTIC MODE: edges only, no ideal arch
            edges_combined = edges_np

        # Distance transform for spatial awareness (normalized to 20px)
        dist = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        dist_norm = np.clip(dist / 20.0, 0, 1) * 255
        dist_norm = dist_norm.astype(np.uint8)

        conditioning = np.stack([
            mask_binary,
            edges_combined,
            dist_norm
        ], axis=-1).astype(np.uint8)

        conditioning_img = Image.fromarray(conditioning, mode="RGB")

        return conditioning_img

    def _generate_ideal_arch(self, mask):
        """
        Generate synthetic ideal dental arch geometry.

        Replaces original tooth structure with perfect veneer alignment
        to override crooked geometry. Uses moderate-strength lines
        (intensity 100-160) with Gaussian softening to guide without
        dictating exact pixel placement.

        Arch line intensities:
        - Separators: 120 (subtle vertical hints between teeth)
        - Arch outline: 140 (parabolic gumline curve)
        - Midline: 100 (horizontal reference)
        - Occlusion plane: 120 (bite plane hint)

        These values are chosen so that after (9,9)/sigma 2.5 blur,
        they produce soft gradients in the 40-80 range that ControlNet
        interprets as gentle guidance rather than hard structure.

        Args:
            mask: PIL Image (tooth mask)

        Returns:
            numpy array: Ideal arch structure (grayscale)
        """
        mask_np = np.array(mask)

        if mask_np.max() == 0:
            return mask_np

        # Find bounding box of teeth region
        coords = np.where(mask_np > 127)
        if len(coords[0]) == 0:
            return mask_np

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        width = x_max - x_min
        height = y_max - y_min

        if width < 10 or height < 10:
            return mask_np

        arch = np.zeros_like(mask_np)

        # --- Symmetric tooth separators (8 upper teeth visible in smile) ---
        # Intensity 120: visible guidance, not blueprint
        # Thickness 1: thin lines that blur into soft gradients
        num_teeth = 8
        tooth_width = width / num_teeth

        for i in range(num_teeth + 1):
            x = int(x_min + i * tooth_width)
            # Parabolic arch: teeth curve upward at edges (natural dental arch)
            t = (i / num_teeth - 0.5) * 2  # -1 to 1
            y_offset = int(height * 0.08 * t * t)
            pt_top = (x, y_min + y_offset)
            pt_bot = (x, y_max - int(height * 0.05))
            cv2.line(arch, pt_top, pt_bot, 120, 1)

        # --- Arch outline (parabolic curve for upper gumline) ---
        # Intensity 140: slightly stronger than separators for arch shape
        arch_points = []
        for i in range(50):
            t_frac = i / 49.0
            x = int(x_min + t_frac * width)
            offset = height * 0.08 * ((t_frac - 0.5) * 2) ** 2
            arch_points.append([x, int(y_min + offset)])
        arch_points = np.array(arch_points, dtype=np.int32)
        cv2.polylines(arch, [arch_points], False, 140, 1)

        # --- Incisal edge line (lower boundary of upper teeth) ---
        # Smooth parabolic curve along the bottom edge
        incisal_points = []
        for i in range(50):
            t_frac = i / 49.0
            x = int(x_min + t_frac * width)
            # Gentle upward curve at edges for natural smile arc
            offset = height * 0.05 * ((t_frac - 0.5) * 2) ** 2
            incisal_points.append([x, int(y_max - int(height * 0.05) - offset)])
        incisal_points = np.array(incisal_points, dtype=np.int32)
        cv2.polylines(arch, [incisal_points], False, 130, 1)

        # --- Horizontal midline reference ---
        # Intensity 100: weakest line, just a spatial anchor
        cy = (y_min + y_max) // 2
        cv2.line(arch, (x_min, cy), (x_max, cy), 100, 1)

        # --- Occlusion plane ---
        # Intensity 120: subtle hint at where upper/lower teeth meet
        # Position at ~52% from top of teeth region
        occlusion_y = int(y_min + height * 0.52)
        cv2.line(arch, (x_min, occlusion_y), (x_max, occlusion_y), 120, 1)

        # Mask the arch to teeth region only (no lip/gum contamination)
        arch[mask_np < 127] = 0

        # Gaussian blur: (9,9) kernel, sigma 2.5
        # Softens hard pixel edges into smooth gradients
        # This is the key balance: visible enough for ControlNet to pick up,
        # soft enough that it doesn't enforce rigid structure
        arch = cv2.GaussianBlur(arch, (9, 9), 2.5)

        return arch

    def _generate_edges(self, image, low_threshold=25, high_threshold=80):
        """
        Generate edge map using Canny detection with adaptive preprocessing.

        Args:
            image: PIL Image
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold

        Returns:
            numpy array: Edge map (grayscale)
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Bilateral filter: reduce noise while preserving structural edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Dilate slightly to ensure connected tooth boundaries
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges