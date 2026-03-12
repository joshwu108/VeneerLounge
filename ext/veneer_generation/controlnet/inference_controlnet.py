"""
ControlNet Inference Pipeline for Veneer Preview Generation.
"""

import argparse
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image
import logging
import cv2
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DPMSolverMultistepScheduler

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'individual_tooth_segmentation'))
from src.network.model import ResNeSt50_TC as TeethSegmentationNet

logger = logging.getLogger(__name__)

class VeneerControlNetGenerator:
    """
    Parameter Presets Guide:

    NATURAL (default):
        - num_inference_steps=28, guidance_scale=9.0
        - controlnet_conditioning_scale=0.65, strength=0.88
        - Best for realistic, high-quality veneer previews

    DRAMATIC:
        - num_inference_steps=38, guidance_scale=10.0
        - controlnet_conditioning_scale=0.75, strength=0.92
        - For maximum whitening and perfect alignment

    CONSERVATIVE:
        - num_inference_steps=25, guidance_scale=7.5
        - controlnet_conditioning_scale=0.55, strength=0.75
        - Minimal changes, natural appearance

    FAST:
        - num_inference_steps=18, guidance_scale=8.5
        - controlnet_conditioning_scale=0.65, strength=0.85
        - Quick preview with good quality
    """

    def __init__(
        self,
        controlnet_path,
        base_model_path="runwayml/stable-diffusion-v1-5",
        segmentation_checkpoint=None,
        device='cuda'
    ):
        """
        Initialize the veneer generator.

        Args:
            controlnet_path: Path to trained ControlNet weights
            base_model_path: Path to base Stable Diffusion model
            segmentation_checkpoint: Path to tooth segmentation checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = self._resolve_device(device)
        dtype = torch.float16 if self.device.type in ('cuda', 'mps') else torch.float32
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None
        )

        # DPM++ SDE with Karras sigmas: better detail retention at low step counts
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True
        )
        self.pipe = self.pipe.to(self.device)

        # Optimize for speed
        if self.device.type in ('cuda', 'mps'):
            if self.device.type == 'cuda':
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    self.pipe.enable_attention_slicing(slice_size="auto")
            else:
                self.pipe.enable_attention_slicing(slice_size="auto")
            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass
            # torch.compile for CUDA only (not yet stable on MPS)
            if self.device.type == 'cuda' and hasattr(torch, 'compile'):
                try:
                    self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                    self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode="reduce-overhead")
                except Exception:
                    pass

        self.seg_model = None
        if segmentation_checkpoint and segmentation_checkpoint != 'None':
            try:
                self.seg_model = TeethSegmentationNet(in_ch=3, out_ch=1)
                checkpoint = torch.load(segmentation_checkpoint, map_location=self.device)

                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'net_state_dict' in checkpoint:
                    state_dict = checkpoint['net_state_dict']
                else:
                    state_dict = checkpoint

                if any(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                self.seg_model.load_state_dict(state_dict)
                self.seg_model.to(self.device)
                self.seg_model.eval()
                print("✓ Tooth segmentation model loaded")
            except Exception as e:
                print(f"Warning: Could not load segmentation model: {e}")
                print("Will use fallback color-based segmentation")
                self.seg_model = None
        else:
            print("Tooth segmentation model not provided - using fallback method")

        print("✓ Veneer generator initialized successfully!")

    @staticmethod
    def _resolve_device(requested):
        """Resolve the best available device: cuda > mps > cpu."""
        if requested == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    @staticmethod
    def get_preset_params(preset='natural'):
        """
        Get predefined parameter sets for different enhancement levels.

        Args:
            preset: 'natural', 'dramatic', 'conservative', or 'fast'

        Returns:
            dict: Parameters for generate_veneer_preview
        """
        presets = {
            'natural': {
                'num_inference_steps': 20,
                'guidance_scale': 6.5,
                'controlnet_conditioning_scale': 0.70,
                'strength': 0.60,
                'whitening_target': 215.0,
                'whitening_strength': 0.45,
                'edge_threshold_low': 25,
                'edge_threshold_high': 80
            },
            'dramatic': {
                'num_inference_steps': 25,
                'guidance_scale': 7.5,
                'controlnet_conditioning_scale': 0.65,
                'strength': 0.70,
                'whitening_target': 220.0,
                'whitening_strength': 0.55,
                'edge_threshold_low': 20,
                'edge_threshold_high': 70
            },
            'conservative': {
                'num_inference_steps': 15,
                'guidance_scale': 5.5,
                'controlnet_conditioning_scale': 0.75,
                'strength': 0.50,
                'whitening_target': 212.0,
                'whitening_strength': 0.35,
                'edge_threshold_low': 25,
                'edge_threshold_high': 90
            },
            'fast': {
                'num_inference_steps': 12,
                'guidance_scale': 5.5,
                'controlnet_conditioning_scale': 0.75,
                'strength': 0.50,
                'whitening_target': 214.0,
                'whitening_strength': 0.42,
                'edge_threshold_low': 25,
                'edge_threshold_high': 80
            },
            'veneers_max': {
                'num_inference_steps': 30,
                'guidance_scale': 8.0,
                'controlnet_conditioning_scale': 0.60,
                'strength': 0.75,
                'whitening_target': 220.0,
                'whitening_strength': 0.60,
                'edge_threshold_low': 20,
                'edge_threshold_high': 70
            }
        }
        return presets.get(preset.lower(), presets['natural'])

    def generate_segmentation_mask(self, image, target_size=(768, 768)):
        """
        Generate tooth segmentation mask.

        Args:
            image: PIL Image
            target_size: Target size for mask

        Returns:
            PIL Image of segmentation mask
        """
        if self.seg_model is None:
            print("Warning: Using fallback color-based segmentation")
            image_resized = image.resize(target_size, Image.LANCZOS)
            img_array = np.array(image_resized)

            # Convert to HSV for better white detection
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

            # Detect white/bright regions (teeth)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)

            return Image.fromarray(mask, mode='L')

        image = image.resize(target_size, Image.LANCZOS)
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            output = self.seg_model(img_tensor)
            mask = torch.sigmoid(output) > 0.5
            mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        return Image.fromarray(mask, mode='L')


    def generate_edge_map(self, image, low_threshold=100, high_threshold=200):
        """
        Generate edge map using Canny detection with adaptive preprocessing.
        Args:
            image: PIL Image
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold

        Returns:
            PIL Image of edge map
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Dilate slightly to ensure connected tooth boundaries
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return Image.fromarray(edges, mode='L')

    def feather_mask(self, mask, feather_px=25):
        binary = (mask > 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist = np.clip(dist / feather_px, 0, 1)
        feathered = (dist * 255).astype(np.uint8)
        return feathered

    def composite_back(self, original, generated_crop, mask, offset):
        x0, y0 = offset
        final = original.copy()
        mask_np = np.array(mask)
        mask_np = cv2.GaussianBlur(mask_np, (9, 9), 0)
        mask = Image.fromarray(mask_np, mode='L')
        final.paste(generated_crop, (x0, y0), mask=mask)
        return final

    def create_conditioning_image(self, image, mask=None, use_edges=True):
        """
        Create conditioning image from segmentation and edges.

        Args:
            image: PIL Image (input smile)
            mask: PIL Image (segmentation mask, optional)
            use_edges: Whether to include edge detection

        Returns:
            PIL Image for ControlNet conditioning
        """
        # Generate mask if not provided
        if mask is None and self.seg_model is not None:
            mask = self.generate_segmentation_mask(image)

        edges = None
        if use_edges:
            edges = self.generate_edge_map(image)
            edges_np = np.array(edges)
            mask_np = np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            edges_np[~mask_np] = 0
            edges = Image.fromarray(edges_np, mode='L')

        # Combine into conditioning image
        if mask is not None and edges is not None:
            # R=mask, G=edges, B=zeros
            mask_array = np.array(mask)
            edges_array = np.array(edges)
            conditioning = np.stack([
                mask_array,
                edges_array,
                np.zeros_like(mask_array)
            ], axis=-1).astype(np.uint8)
        elif mask is not None:
            # Just mask in all channels
            mask_array = np.array(mask)
            conditioning = np.stack([mask_array] * 3, axis=-1).astype(np.uint8)
        else:
            raise ValueError("Must provide either mask or segmentation model")

        return Image.fromarray(conditioning)

    def refine_tooth_mask(self, mask, erosion_px=3):
        """
        Refine tooth mask with contour smoothing and morphological operations.
        Removes small artifacts, smooths boundaries via polygon approximation,
        and erodes conservatively to avoid gum bleeding.
        """
        mask_np = np.array(mask)

        # 1. Morphological close to fill internal holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        refined = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 2. Find contours and approximate with smooth polygons
        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoothed = np.zeros_like(refined)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Drop small noise blobs
                continue
            # Approximate contour to remove jaggedness
            epsilon = 0.008 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(smoothed, [approx], -1, 255, cv2.FILLED)

        # 3. Gaussian blur then re-threshold for sub-pixel smooth edges
        smoothed = cv2.GaussianBlur(smoothed, (9, 9), 3)
        _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)

        # 4. Conservative erosion to pull away from gumline
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px, erosion_px))
        smoothed = cv2.erode(smoothed, kernel_erode, iterations=1)

        return Image.fromarray(smoothed, mode='L')

    def advanced_feather_mask(self, mask, inner_feather_px=6, outer_feather_px=18):
        """
        Distance-based feathering with smooth boundary gradient.
        Produces a mask that is 1.0 in the interior and smoothly falls to 0 at edges.
        """
        mask_np = (np.array(mask) > 0).astype(np.uint8) * 255

        # Inward distance for soft edge falloff
        dist_in = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
        feathered = np.clip(dist_in / inner_feather_px, 0, 1)

        # Smooth the boundary gradient
        feathered = cv2.GaussianBlur(feathered.astype(np.float32), (15, 15), 0)

        return (feathered * 255).astype(np.uint8)

    def poisson_blend(self, generated_crop, original_crop, mask, use_mixed=False):
        """
        Seamless cloning using Poisson editing for gradient-domain compositing.

        Args:
            generated_crop: Generated teeth region (numpy array)
            original_crop: Original image crop (numpy array)
            mask: Binary mask (numpy array)
            use_mixed: Use MIXED_CLONE for better texture preservation

        Returns:
            Blended image (numpy array)
        """
        # Ensure mask is binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Erode mask slightly to avoid edge artifacts
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=1)

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

    def generate_ideal_arch_conditioning(self, mask, image_size=768):
        """
        Generate a synthetic ideal dental arch to replace original tooth structure
        in the conditioning image. Forces diffusion toward perfect veneer alignment
        instead of inheriting the original crooked tooth geometry.
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

        # Draw symmetric tooth separators (8 upper teeth visible in smile)
        num_teeth = 8
        tooth_width = width / num_teeth

        for i in range(num_teeth + 1):
            x = int(x_min + i * tooth_width)
            # Slight parabolic arch: teeth curve upward at edges
            t = (i / num_teeth - 0.5) * 2  # -1 to 1
            y_offset = int(height * 0.08 * t * t)  # Gentle parabola
            pt_top = (x, y_min + y_offset)
            pt_bot = (x, y_max - int(height * 0.05))
            cv2.line(arch, pt_top, pt_bot, 120, 1)

        # Draw arch outline (parabolic curve for upper gumline)
        arch_points = []
        for i in range(50):
            t_frac = i / 49.0
            x = int(x_min + t_frac * width)
            offset = height * 0.08 * ((t_frac - 0.5) * 2) ** 2
            arch_points.append([x, int(y_min + offset)])
        arch_points = np.array(arch_points, dtype=np.int32)
        cv2.polylines(arch, [arch_points], False, 140, 1)

        # Horizontal midline
        cy = (y_min + y_max) // 2
        cv2.line(arch, (x_min, cy), (x_max, cy), 100, 1)

        # Occlusion plane - subtle hint, not enforcement
        occlusion_y = int(y_min + height * 0.52)
        cv2.line(arch, (x_min, occlusion_y), (x_max, occlusion_y), 120, 1)

        # Mask the arch to teeth region only
        arch[mask_np < 127] = 0

        # Gaussian blur: (9,9)/sigma 2.5 for smooth gradients
        arch = cv2.GaussianBlur(arch, (9, 9), 2.5)

        return arch

    def add_enamel_texture(self, image_np, mask_bool, strength=0.35):
        """
        Add realistic enamel micro-texture using multi-frequency noise.
        Produces subtle vertical ridges and translucency variation near edges,
        replacing uniform Gaussian noise with structured porcelain-like texture.
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

    def generate_veneer_preview(
        self,
        image,
        prompt=None,
        negative_prompt=None,
        num_inference_steps=34,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.45,
        strength=0.82,
        seed=None,
        debug_dir=None,
        bounding_box=None,
        edge_threshold_low=25,
        edge_threshold_high=80,
        whitening_target=215.0,
        whitening_strength=0.45,
        use_poisson_blend=True
    ):
        """
        Generate veneer preview for an input smile image.

        Args:
            image: PIL Image or path to image
            prompt: Text prompt (optional, uses default if None)
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps (default: 22)
            guidance_scale: Classifier-free guidance (default: 7.5, range: 5.0-9.0)
            controlnet_conditioning_scale: Structural guidance strength (default: 0.35, range: 0.2-0.5)
            strength: Modification strength (default: 0.78, range: 0.6-0.9)
            seed: Random seed for reproducibility
            debug_dir: Directory to save debug images (optional)
            bounding_box: Bounding box for mouth region
            edge_threshold_low: Canny low threshold (default: 30)
            edge_threshold_high: Canny high threshold (default: 100)
            whitening_target: Target brightness in LAB space (default: 212.0, range: 200-220)
            whitening_strength: Whitening boost factor (default: 0.45, range: 0.2-0.6)
            use_poisson_blend: Use Poisson blending for seamless compositing (default: True)
        Returns:
            PIL Image of veneer preview
        """
        # Load image if path
        GEN_SIZE = 512
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        output_image = image.copy()
        orig_w, orig_h = output_image.size

        # CORRECT APPROACH: Apply percentage coordinates to ORIGINAL image first
        # Default bounding box: mouth area (approximately 45-75% vertically for portrait/landscape)
        # This targets the lower-middle region where the mouth typically appears
        x1_pct, y1_pct, x2_pct, y2_pct = 0, 45, 100, 75

        if bounding_box:
            # Extract bounding box as percentages
            x1_pct = bounding_box['x']
            y1_pct = bounding_box['y']
            x2_pct = bounding_box['x'] + bounding_box['width']
            y2_pct = bounding_box['y'] + bounding_box['height']

        # Calculate bounding box in ORIGINAL image coordinates
        x1_orig = int(x1_pct / 100.0 * orig_w)
        y1_orig = int(y1_pct / 100.0 * orig_h)
        x2_orig = int(x2_pct / 100.0 * orig_w)
        y2_orig = int(y2_pct / 100.0 * orig_h)
        x1_orig = max(0, min(orig_w, x1_orig))
        y1_orig = max(0, min(orig_h, y1_orig))
        x2_orig = max(0, min(orig_w, x2_orig))
        y2_orig = max(0, min(orig_h, y2_orig))

        # Crop from ORIGINAL image
        crop_orig = image.crop((x1_orig, y1_orig, x2_orig, y2_orig))

        # Resize crop to GEN_SIZE for processing
        image = crop_orig.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)

        # Try to generate tooth mask via segmentation
        tooth_mask = self.generate_segmentation_mask(image)
        tooth_mask_np = np.array(tooth_mask)

        # If segmentation failed (mask is mostly empty), create a simple center rectangle mask
        if tooth_mask_np.mean() < 10:  # Very little white detected
            tooth_mask_np = np.zeros((GEN_SIZE, GEN_SIZE), dtype=np.uint8)
            h_start, h_end = int(GEN_SIZE * 0.3), int(GEN_SIZE * 0.7)
            w_start, w_end = int(GEN_SIZE * 0.15), int(GEN_SIZE * 0.85)
            tooth_mask_np[h_start:h_end, w_start:w_end] = 255
            tooth_mask = Image.fromarray(tooth_mask_np, mode="L")
        else:
            # Contour-smoothed refinement with conservative erosion
            tooth_mask = self.refine_tooth_mask(tooth_mask, erosion_px=3)

        mask_np = np.array(tooth_mask)
        h, w = mask_np.shape

        # Distance-based feathering for smooth boundary gradient
        mask_np = self.advanced_feather_mask(tooth_mask, inner_feather_px=8, outer_feather_px=30)
        tooth_mask = Image.fromarray(mask_np, mode="L")

        if debug_dir:
            tooth_mask.save(Path(debug_dir) / "gated_tooth_mask.png")
            print(f"Debug: Mask stats - min={mask_np.min()}, max={mask_np.max()}, mean={mask_np.mean():.1f}")

        # Store offset in ORIGINAL image space for final compositing (already calculated above)
        offset_orig = (x1_orig, y1_orig)
        crop_img, crop_mask = image, tooth_mask

        sd_crop_img = crop_img.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)
        sd_crop_mask = crop_mask.resize((GEN_SIZE, GEN_SIZE), Image.NEAREST)  # NEAREST for masks

        mask_np = np.array(sd_crop_mask)
        # Erosion already done in refine_tooth_mask — no additional erosion here
        # to avoid shrinking the mask below the actual tooth area

        # Create binary mask for operations
        mask_bool = mask_np > 127

        # --- Synthetic ideal-arch conditioning ---
        # Replace original tooth edge structure with canonical arch geometry
        # to force diffusion toward perfect veneer alignment
        ideal_arch = self.generate_ideal_arch_conditioning(sd_crop_mask)

        # Generate edges for OUTSIDE mask only (preserve lips, face structure)
        edges = self.generate_edge_map(sd_crop_img, low_threshold=edge_threshold_low, high_threshold=edge_threshold_high)
        edges_np = np.array(edges)

        # Swap: inside mask = synthetic arch, outside mask = real edges
        # Use np.where to safely handle potential shape mismatches
        if ideal_arch.shape != edges_np.shape:
            ideal_arch = cv2.resize(ideal_arch, (edges_np.shape[1], edges_np.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        edges_np = np.where(mask_bool, ideal_arch, edges_np)

        # Calculate distance transform for spatial awareness
        mask_bin = mask_bool.astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
        dist_norm = np.clip(dist_transform / 20.0, 0, 1) * 255
        dist_norm = dist_norm.astype(np.uint8)

        # Multi-channel conditioning: R=mask, G=ideal_arch+edges, B=distance
        conditioning = np.stack([mask_np, edges_np, dist_norm], axis=-1)
        conditioning_image = Image.fromarray(conditioning)

        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            tooth_mask.save(debug_path / 'tooth_mask.png')
            image.save(debug_path / 'input_resized.png')
            conditioning_image.save(debug_path / 'conditioning_edges.png')
            # Save ideal arch for debugging
            Image.fromarray(ideal_arch, mode='L').save(debug_path / 'ideal_arch.png')

        # --- Prompts condensed to ≤ 77 CLIP tokens (was 154 → silently truncated) ---
        if prompt is None:
            prompt = (
                "(perfect porcelain veneers:1.4), "
                "(symmetrical dental arch:1.3), "
                "uniform white teeth, closed bite, ideal alignment, "
                "smooth enamel surface, natural translucency, "
                "professional dental photography, 85mm macro lens, "
                "studio lighting, photorealistic"
            )
        if negative_prompt is None:
            negative_prompt = (
                "crooked teeth, misaligned, gaps, yellow stains, "
                "blue tint, plastic, CGI, overexposed, blurry, "
                "cartoon, open bite, distorted gums"
            )

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Clear CUDA cache before generation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sd_crop_img,
            mask_image=sd_crop_mask,
            control_image=conditioning_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            generator=generator
        )

        # Clear cache after generation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        generated_sd = output.images[0]

        # Resize generated image back to the original bounding box dimensions
        # bbox dimensions already calculated from original coordinates
        bbox_w, bbox_h = x2_orig - x1_orig, y2_orig - y1_orig
        generated_crop = generated_sd.resize((bbox_w, bbox_h), Image.LANCZOS)

        # --- Enamel micro-texture (replaces uniform Gaussian noise) ---
        match_mask = sd_crop_mask.resize(generated_crop.size, Image.NEAREST)
        mask_bool_final = np.array(match_mask) > 127
        gen_np = np.array(generated_crop)
        gen_np = self.add_enamel_texture(gen_np, mask_bool_final, strength=0.35)

        # --- Clamped LAB whitening (no bloom, no blue tint) ---
        lab = cv2.cvtColor(gen_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[..., 0].astype(np.float32)
        a_channel = lab[..., 1].astype(np.float32)
        b_channel = lab[..., 2].astype(np.float32)

        if np.any(mask_bool_final):
            current_mean = l_channel[mask_bool_final].mean()
            target_l = whitening_target

            # Proportional boost capped at L=230 to prevent bloom
            boost = (target_l - current_mean) * whitening_strength
            l_channel[mask_bool_final] = np.clip(
                l_channel[mask_bool_final] + boost,
                current_mean * 0.9,  # Don't darken any pixel below 90% of current mean
                230.0                 # Hard ceiling prevents blown highlights
            )

            # Nudge chromaticity toward neutral (a=128, b=128 in OpenCV LAB 0-255 scale)
            # Pull 35% toward neutral, preserving 65% of original tone
            a_neutral = 128.0
            b_neutral = 128.0
            a_channel[mask_bool_final] = a_channel[mask_bool_final] * 0.65 + a_neutral * 0.35
            b_channel[mask_bool_final] = b_channel[mask_bool_final] * 0.65 + b_neutral * 0.35

            # Anti-bloom: soft-clamp highlights above L=228
            hot_pixels = mask_bool_final & (l_channel > 228)
            if np.any(hot_pixels):
                excess = l_channel[hot_pixels] - 228
                l_channel[hot_pixels] = 228 + excess * 0.25  # Aggressive highlight compression

        lab[..., 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
        lab[..., 1] = np.clip(a_channel, 0, 255).astype(np.uint8)
        lab[..., 2] = np.clip(b_channel, 0, 255).astype(np.uint8)
        generated_crop = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

        # --- Multi-stage blending for seamless compositing ---
        orig_crop = np.array(output_image.crop((x1_orig, y1_orig, x2_orig, y2_orig)))
        gen_crop = np.array(generated_crop)

        if use_poisson_blend:
            # Try Poisson blending first for gradient-domain compositing
            final_mask_poisson = sd_crop_mask.resize((bbox_w, bbox_h), Image.NEAREST)
            mask_poisson = np.array(final_mask_poisson)

            poisson_result = self.poisson_blend(gen_crop, orig_crop, mask_poisson, use_mixed=False)

            if poisson_result is not None:
                output_image.paste(Image.fromarray(poisson_result), offset_orig)
                if debug_dir:
                    output_image.save(Path(debug_dir) / 'output.png')
                return output_image

        # Fall back to advanced alpha blending with histogram matching
        final_mask = sd_crop_mask.resize((bbox_w, bbox_h), Image.LANCZOS)
        mask_np = np.array(final_mask).astype(np.float32) / 255.0

        # Distance-based feathering with narrower zone (less halo)
        mask_binary = (mask_np > 0.05).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)

        feather_width = 20  # Narrower than 45 to reduce halo effect
        smooth_mask = np.clip(dist_transform / feather_width, 0, 1)

        # Edge-aware bilateral filter preserves gumline sharpness
        smooth_mask_8u = (smooth_mask * 255).astype(np.uint8)
        try:
            # Try guided filter if opencv-contrib is available
            orig_gray = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
            smooth_mask_8u = cv2.ximgproc.guidedFilter(
                guide=orig_gray, src=smooth_mask_8u, radius=8, eps=100
            )
        except AttributeError:
            # Fall back to bilateral filter (standard opencv)
            smooth_mask_8u = cv2.bilateralFilter(smooth_mask_8u, d=15, sigmaColor=30, sigmaSpace=15)
        smooth_mask = smooth_mask_8u.astype(np.float32) / 255.0

        # No gamma correction - linear falloff is more natural at gumline

        orig_crop_f = orig_crop.astype(np.float32)
        gen_crop_f = gen_crop.astype(np.float32)

        # Color and brightness matching in transition zones
        alpha_3d = smooth_mask[..., np.newaxis]

        # Define transition zone (where blending happens)
        transition_zone = (alpha_3d[..., 0] > 0.02) & (alpha_3d[..., 0] < 0.98)

        if np.any(transition_zone):
            # Match histogram in LAB space for perceptually correct blending
            gen_lab = cv2.cvtColor(gen_crop.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            orig_lab = cv2.cvtColor(orig_crop.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

            # Get boundary pixels (outer transition zone)
            outer_boundary = (alpha_3d[..., 0] > 0.02) & (alpha_3d[..., 0] < 0.3)

            if np.any(outer_boundary):
                # Match L (brightness) channel at boundaries
                for c in range(3):
                    gen_boundary = gen_lab[..., c][outer_boundary]
                    orig_boundary = orig_lab[..., c][outer_boundary]

                    if len(gen_boundary) > 10:
                        # Calculate mean and std
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
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        output_image.paste(Image.fromarray(blended), offset_orig)

        if debug_dir:
            output_image.save(Path(debug_dir) / 'output.png')
        return output_image

    def match_color(self, source, target):
        src = np.array(source).astype(np.float32)
        tgt = np.array(target).astype(np.float32)
        for c in range(3):
            src_mean, src_std = src[..., c].mean(), src[..., c].std()
            tgt_mean, tgt_std = tgt[..., c].mean(), tgt[..., c].std()
            src[..., c] = (src[..., c] - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean

        return Image.fromarray(np.clip(src, 0, 255).astype(np.uint8))

    def generate_comparison(self, image, output_path=None, **kwargs):
        """
        Generate side-by-side comparison of before/after.

        Args:
            image: Input image
            output_path: Path to save comparison (optional)
            **kwargs: Arguments for generate_veneer_preview

        Returns:
            PIL Image of comparison
        """
        # Load image
        GEN_SIZE = 512
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        image = image.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)
        preview = self.generate_veneer_preview(image, **kwargs)
        comparison = Image.new('RGB', (2 * GEN_SIZE, GEN_SIZE))
        comparison.paste(image, (0, 0))
        comparison.paste(preview, (GEN_SIZE, 0))
        # Save if requested
        if output_path:
            comparison.save(output_path, quality=95)
            print(f"✓ Comparison saved to {output_path}")

        return comparison


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Generate veneer preview using ControlNet'
    )
    parser.add_argument(
        '--controlnet',
        type=str,
        required=True,
        help='Path to trained ControlNet model'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='runwayml/stable-diffusion-v1-5',
        help='Base Stable Diffusion model'
    )
    parser.add_argument(
        '--segmentation',
        type=str,
        default='../../individual_tooth_segmentation/checkpoints/CP_teeth_seg.pth',
        help='Path to segmentation model checkpoint'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Input smile image'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for veneer preview'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Custom text prompt'
    )
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default=None,
        help='Negative prompt'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=28,
        help='Number of inference steps'
    )
    parser.add_argument(
        '--guidance',
        type=float,
        default=9.0,
        help='Guidance scale'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Generate side-by-side comparison'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()
    generator = VeneerControlNetGenerator(
        controlnet_path=args.controlnet,
        base_model_path=args.base_model,
        segmentation_checkpoint=args.segmentation,
        device=args.device
    )

    if args.comparison:
        result = generator.generate_comparison(
            image=args.image,
            output_path=args.output,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
    else:
        image = Image.open(args.image).convert('RGB')
        result = generator.generate_veneer_preview(
            image=image,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
        result.save(args.output, quality=95)
        print(f"✓ Veneer preview saved to {args.output}")


if __name__ == "__main__":
    main()