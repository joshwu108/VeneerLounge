"""
Unified Veneer Preview Service

This service provides a single interface for veneer preview generation with
- SDXL (recommended - best quality, preserves identity)
- ControlNet (legacy - geometric + color changes)
- Pix2pix (fast baseline for cosmetic changes only)

The service handles:
1. Model loading and caching
2. Image preprocessing
3. Inference
4. Result post-processing
5. Base64 encoding for API responses
"""
import base64
import io
import sys
import threading
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import cv2
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'ext' / 'veneer_generation'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'ext' / 'individual_tooth_segmentation'))

# Segmentation model checkpoint path
_SEG_CHECKPOINT = Path(__file__).parent.parent.parent / 'ext' / 'individual_tooth_segmentation' / 'checkpoints' / 'CP_teeth_seg.pth'


class VeneerPreviewService:
    """
    Unified service for veneer preview generation.
    Supports multiple model backends with a consistent interface.
    """

    def __init__(self, model_type='sdxl', **model_config):
        """
        Initialize the veneer preview service.

        Args:
            model_type: Type of model to use ('sdxl', 'controlnet')
            **model_config: Model-specific configuration
                For SDXL:
                    - No required config, uses stabilityai/stable-diffusion-xl-base-1.0
                For ControlNet:
                    - controlnet_path: Path to ControlNet weights
                    - base_model_path: Path to Stable Diffusion base
                    - segmentation_checkpoint: Path to segmentation model
        """
        self.model_type = model_type
        self.generator = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._pipeline_lock = threading.Lock()

        self.seg_model = None

        print(f"Initializing Veneer Preview Service...")
        print(f"  Model type: {model_type}")
        print(f"  Device: {self.device}")

        # Load tooth segmentation model for mask generation
        self._load_segmentation_model()

        # Initialize the appropriate model
        self._initialize_generator(model_config)

    def _initialize_generator(self, config):
        """
        Initialize the veneer generator based on model type.

        Args:
            config: Model-specific configuration dictionary
        """
        if self.model_type == 'sdxl':
            self._init_sdxl(config)
        elif self.model_type == 'controlnet':
            self._init_controlnet(config)
        elif self.model_type == 'pix2pix':
            self._init_pix2pix(config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_segmentation_model(self):
        """Load the ResNeSt50 tooth segmentation model for precise mask generation."""
        if not _SEG_CHECKPOINT.exists():
            print(f"⚠ Segmentation checkpoint not found at {_SEG_CHECKPOINT}, falling back to HSV")
            return

        try:
            from src.network.model import ResNeSt50_TC

            self.seg_model = ResNeSt50_TC(in_ch=3, out_ch=1)
            checkpoint = torch.load(str(_SEG_CHECKPOINT), map_location=self.device)

            # Handle multiple checkpoint formats (same as inference_controlnet.py)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'net_state_dict' in checkpoint:
                state_dict = checkpoint['net_state_dict']
            else:
                state_dict = checkpoint

            # Strip DataParallel 'module.' prefix if present
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.seg_model.load_state_dict(state_dict)
            self.seg_model.to(self.device)
            self.seg_model.eval()
            print("✓ Tooth segmentation model loaded")
        except Exception as e:
            print(f"⚠ Failed to load segmentation model: {e}, falling back to HSV")
            self.seg_model = None

    def _init_sdxl(self, config):
        """Initialize SDXL inpainting generator (recommended)."""
        from controlnet.veneer_generator_simple import SimpleVeneerGenerator

        self.generator = SimpleVeneerGenerator(device=str(self.device))
        print("✓ SDXL generator initialized")

    def _init_controlnet(self, config):
        """Initialize ControlNet generator."""
        from controlnet.inference_controlnet import VeneerControlNetGenerator

        # Only controlnet_path is required, segmentation is optional
        if 'controlnet_path' not in config:
            raise ValueError(f"Missing required config key for ControlNet: controlnet_path")

        self.generator = VeneerControlNetGenerator(
            controlnet_path=config['controlnet_path'],
            base_model_path=config.get('base_model_path', 'runwayml/stable-diffusion-v1-5'),
            segmentation_checkpoint=config.get('segmentation_checkpoint', None),
            device=str(self.device)
        )

        print("✓ ControlNet generator initialized")

    def _init_pix2pix(self, config):
        """Initialize Pix2pix generator."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                              'ext/veneer_generation/pix2pix'))
        from inference_pix2pix import VeneerPix2PixGenerator

        required_keys = ['checkpoint_path']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key for Pix2pix: {key}")

        self.generator = VeneerPix2PixGenerator(
            checkpoint_path=config['checkpoint_path'],
            device=str(self.device)
        )

        print("✓ Pix2pix generator initialized")

    def generate_from_pil(
        self,
        image,
        intensity=0.8,
        preserve_geometry=False,
        custom_prompt=None,
        bounding_box=None,
        **kwargs
    ):
        """
        Generate veneer preview from PIL Image.

        Args:
            image: PIL Image
            intensity: Transformation intensity (0-1)
                Higher = more dramatic changes
            preserve_geometry: If True, minimize tooth movement
            custom_prompt: Custom text prompt (ControlNet only)
            **kwargs: Additional model-specific arguments

        Returns:
            PIL Image of veneer preview
        """
        if self.model_type == 'sdxl':
            return self._generate_sdxl(
                image, intensity, bounding_box, **kwargs
            )
        elif self.model_type == 'controlnet':
            return self._generate_controlnet(
                image, intensity, preserve_geometry, custom_prompt, bounding_box, **kwargs
            )
        elif self.model_type == 'pix2pix':
            return self._generate_pix2pix(image, intensity, **kwargs)

    def _create_mouth_mask(self, image, bounding_box=None):
        """
        Create a mask for the mouth/teeth region using the neural network
        segmentation model (ResNeSt50). Falls back to HSV thresholding if
        the segmentation model is not available.

        Args:
            image: PIL Image
            bounding_box: Optional dict with x, y, width, height (percentages)

        Returns:
            PIL Image mask (white = teeth area)
        """
        w, h = image.size

        if bounding_box:
            x = int(bounding_box['x'] / 100.0 * w)
            y = int(bounding_box['y'] / 100.0 * h)
            bw = int(bounding_box['width'] / 100.0 * w)
            bh = int(bounding_box['height'] / 100.0 * h)
        else:
            x = int(w * 0.25)
            y = int(h * 0.50)
            bw = int(w * 0.50)
            bh = int(h * 0.30)

        # Expand bounding box vertically to ensure both upper and lower
        # teeth are captured. Add 30% padding above and 40% below since
        # users tend to draw boxes around the upper teeth only.
        pad_top = int(bh * 0.3)
        pad_bottom = int(bh * 0.4)
        y = y - pad_top
        bh = bh + pad_top + pad_bottom

        # Clamp to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        img_np = np.array(image)
        crop = img_np[y:y+bh, x:x+bw]

        if self.seg_model is not None:
            teeth_mask_crop = self._segment_teeth_nn(crop)
            print("  Mask generated via neural network segmentation")
        else:
            teeth_mask_crop = self._segment_teeth_hsv(crop)
            print("  Mask generated via HSV fallback")

        # Place teeth mask into full-size mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y:y+bh, x:x+bw] = teeth_mask_crop

        # Light dilation so SDXL has room to blend at edges
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

        # Save debug mask
        debug_dir = Path(__file__).parent.parent.parent / 'debug_outputs'
        debug_dir.mkdir(exist_ok=True)
        Image.fromarray(mask, mode='L').save(debug_dir / 'auto_mask.png')

        return Image.fromarray(mask, mode='L')

    def _segment_teeth_nn(self, crop_np):
        """
        Run neural network tooth segmentation on a cropped mouth region.

        Args:
            crop_np: numpy array (H, W, 3) RGB crop of mouth region

        Returns:
            numpy array (H, W) uint8 binary mask (255 = teeth)
        """
        crop_h, crop_w = crop_np.shape[:2]

        # Pad to multiple of 32 for the model
        pad_h = (32 - crop_h % 32) % 32
        pad_w = (32 - crop_w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            crop_padded = np.pad(
                crop_np,
                ((pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2),
                 (0, 0)),
                mode='reflect'
            )
        else:
            crop_padded = crop_np

        # Preprocess: HWC -> CHW, normalize to [0, 1]
        img_tensor = torch.from_numpy(crop_padded).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            output = self.seg_model(img_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            ph = pad_h // 2
            pw = pad_w // 2
            prob = prob[ph:ph + crop_h, pw:pw + crop_w]

        # Binary threshold
        mask = (prob > 0.5).astype(np.uint8) * 255

        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return mask

    def _segment_teeth_hsv(self, crop_np):
        """
        Fallback HSV-based teeth detection (used when segmentation model unavailable).

        Args:
            crop_np: numpy array (H, W, 3) RGB crop of mouth region

        Returns:
            numpy array (H, W) uint8 binary mask (255 = teeth)
        """
        crop_h, crop_w = crop_np.shape[:2]
        crop_hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)

        lower = np.array([0, 0, 150])
        upper = np.array([180, 60, 255])
        teeth_mask = cv2.inRange(crop_hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Ellipse fallback if very few teeth pixels detected
        teeth_pixels = np.sum(teeth_mask > 0)
        total_pixels = crop_w * crop_h
        if total_pixels > 0 and teeth_pixels / total_pixels < 0.05:
            teeth_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            center = (crop_w // 2, crop_h // 2)
            axes = (crop_w // 2, crop_h // 2)
            cv2.ellipse(teeth_mask, center, axes, 0, 0, 360, 255, -1)

        return teeth_mask

    def _generate_sdxl(self, image, intensity, bounding_box, **kwargs):
        """
        Generate veneer preview using SDXL inpainting.
        After SDXL generates teeth, composites the result back onto the
        original photo so only the teeth region changes — no rectangular
        boundary artifacts.

        Saves debug images at each step to debug_outputs/sdxl_pipeline/.

        Args:
            image: PIL Image
            intensity: Controls strength (0-1)
            bounding_box: Mouth region coordinates
            **kwargs: Additional arguments

        Returns:
            PIL Image with veneer preview
        """
        import time
        debug_dir = Path(__file__).parent.parent.parent / 'debug_outputs' / 'sdxl_pipeline'
        debug_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # Step 1: Save input
        image.save(debug_dir / '01_input.png')
        print(f"  [debug] 01_input saved ({image.size[0]}x{image.size[1]})")

        # Step 2: Create mask from bounding box (teeth detection)
        mask = self._create_mouth_mask(image, bounding_box)
        mask.save(debug_dir / '02_teeth_mask.png')
        mask_np = np.array(mask)
        pct = np.sum(mask_np > 0) / mask_np.size * 100
        print(f"  [debug] 02_teeth_mask saved (coverage: {pct:.1f}%, min={mask_np.min()}, max={mask_np.max()})")

        # Step 3: Save mask overlay on input for visual check
        overlay = np.array(image).copy()
        overlay[mask_np > 0] = [255, 0, 0]  # Red overlay on teeth
        blended_overlay = (np.array(image).astype(float) * 0.6 + overlay.astype(float) * 0.4).astype(np.uint8)
        Image.fromarray(blended_overlay).save(debug_dir / '03_mask_overlay.png')
        print(f"  [debug] 03_mask_overlay saved")

        # Step 4: Run SDXL inpainting (lock prevents concurrent scheduler corruption)
        strength = intensity
        t1 = time.time()
        print(f"  [debug] Starting SDXL inpainting (strength={strength})...")
        with self._pipeline_lock:
            result = self.generator.generate_from_pil(
                image=image,
                mask=mask,
                strength=strength,
                guidance_scale=kwargs.get('guidance_scale', 7.5),
                num_inference_steps=kwargs.get('steps', 30),
                seed=kwargs.get('seed', None)
            )
        t2 = time.time()
        result.save(debug_dir / '04_sdxl_raw_output.png')
        print(f"  [debug] 04_sdxl_raw_output saved ({t2 - t1:.1f}s)")

        # Step 5: Alpha-composite onto original
        final = self._composite_result(image, result, mask)
        final.save(debug_dir / '05_final_composited.png')
        print(f"  [debug] 05_final_composited saved (total: {time.time() - t0:.1f}s)")

        return final

    def _composite_result(self, original, generated, mask):
        """
        Blend the SDXL-generated image onto the original using a feathered
        version of the teeth mask. This prevents rectangular boundary
        artifacts by ensuring only the teeth region is replaced.

        Args:
            original: PIL Image (original photo)
            generated: PIL Image (SDXL output)
            mask: PIL Image (L mode, teeth mask)

        Returns:
            PIL Image with seamless blend
        """
        # Feather the mask for smooth transition at boundaries
        mask_np = np.array(mask).astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(mask_np, (31, 31), 0)
        alpha = np.stack([alpha] * 3, axis=-1)

        orig_np = np.array(original).astype(np.float32)
        gen_np = np.array(generated).astype(np.float32)

        # Alpha blend: keep original everywhere except teeth
        blended = orig_np * (1.0 - alpha) + gen_np * alpha
        return Image.fromarray(blended.astype(np.uint8))

    def _generate_controlnet(
        self,
        image,
        intensity,
        preserve_geometry,
        custom_prompt,
        bounding_box,
        **kwargs
    ):
        # Use recommended defaults from inference_controlnet.py
        # Allow override via kwargs, otherwise use tuned defaults
        controlnet_scale = kwargs.get('controlnet_conditioning_scale', 0.45)
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        steps = kwargs.get('steps', 30)
        strength = kwargs.get('strength', 0.80)

        if custom_prompt is None:
            if preserve_geometry:
                prompt = """ultra realistic dental veneers, natural enamel translucency, preserved tooth alignment,
                            consistent lighting, subtle surface microtexture, photorealistic dentistry,
                            no change to lips, gums, face, skin, or jaw, teeth only"""
            else:
                prompt = """ultra realistic dental veneers, natural enamel translucency, preserved tooth alignment,
                            consistent lighting, subtle surface microtexture, photorealistic dentistry,
                            no change to lips, gums, face, skin, or jaw, teeth only"""
        else:
            prompt = custom_prompt

        # Enable debug directory
        import os
        debug_dir = Path(__file__).parent.parent.parent / 'debug_outputs'
        os.makedirs(debug_dir, exist_ok=True)

        result = self.generator.generate_veneer_preview(
            image=image,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            strength=strength,
            seed=kwargs.get('seed', None),
            debug_dir=str(debug_dir),
            bounding_box=bounding_box
        )

        return result

    def _generate_pix2pix(self, image, intensity, **kwargs):
        """
        Unimplemented for now
        """
        result = self.generator.generate(image, return_pil=True)
        return result

    def generate_from_base64(
        self,
        base64_image,
        intensity=0.8,
        preserve_geometry=False,
        custom_prompt=None,
        bounding_box=None,
        return_format='base64',
        **kwargs
    ):
        """
        Generate veneer preview from base64 encoded image.
        Args:
            base64_image: Base64 encoded image string
            intensity: Transformation intensity (0-1)
            preserve_geometry: If True, minimize tooth movement
            custom_prompt: Custom text prompt
            return_format: 'base64' or 'pil'
            **kwargs: Additional generation arguments

        Returns:
            Base64 encoded preview image (if return_format='base64')
            or PIL Image (if return_format='pil')
        """
        try:
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]

            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')

            import os
            debug_dir = Path(__file__).parent.parent.parent / 'debug_outputs'
            os.makedirs(debug_dir, exist_ok=True)
            image.save(debug_dir / 'input_image.jpg')
            preview = self.generate_from_pil(
                image=image,
                intensity=intensity,
                preserve_geometry=preserve_geometry,
                custom_prompt=custom_prompt,
                bounding_box=bounding_box,
                **kwargs
            )

            if return_format == 'pil':
                return preview
            buffer = io.BytesIO()
            preview.save(buffer, format='PNG', optimize=False)
            preview_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return f"data:image/png;base64,{preview_base64}"

        except Exception as e:
            raise Exception(f"Error generating veneer preview: {str(e)}")

    def generate_from_file(
        self,
        file_path,
        output_path=None,
        intensity=0.8,
        preserve_geometry=False,
        **kwargs
    ):
        """
        Generate veneer preview from file path.

        Args:
            file_path: Path to input image
            output_path: Path to save output (optional)
            intensity: Transformation intensity
            preserve_geometry: Whether to preserve tooth positions
            **kwargs: Additional generation arguments

        Returns:
            PIL Image of preview
        """
        image = Image.open(file_path).convert('RGB')
        preview = self.generate_from_pil(
            image=image,
            intensity=intensity,
            preserve_geometry=preserve_geometry,
            **kwargs
        )
        if output_path:
            preview.save(output_path, quality=95)
            print(f"✓ Preview saved to {output_path}")

        return preview

_veneer_service = None


def get_veneer_service(model_type='sdxl', force_reload=False, **config):
    """
    Get singleton instance of veneer service.

    Args:
        model_type: Type of model ('sdxl', 'controlnet', or 'pix2pix')
        force_reload: Force reload of service
        **config: Model configuration

    Returns:
        VeneerPreviewService instance
    """
    global _veneer_service

    if _veneer_service is None or force_reload:
        _veneer_service = VeneerPreviewService(
            model_type=model_type,
            **config
        )

    return _veneer_service


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--model', type=str, default='sdxl', choices=['sdxl', 'controlnet', 'pix2pix'])
    parser.add_argument('--controlnet-path', type=str, help='Path to ControlNet weights')
    parser.add_argument('--segmentation', type=str, help='Path to segmentation checkpoint')
    parser.add_argument('--intensity', type=float, default=0.8, help='Transformation intensity')
    parser.add_argument('--preserve-geometry', action='store_true', help='Preserve tooth positions')

    args = parser.parse_args()

    # Configure service
    if args.model == 'sdxl':
        config = {}  # SDXL needs no special config
    elif args.model == 'controlnet':
        config = {
            'controlnet_path': args.controlnet_path,
            'segmentation_checkpoint': args.segmentation
        }
    else:
        config = {}

    service = get_veneer_service(model_type=args.model, **config)
    result = service.generate_from_file(
        file_path=args.image,
        output_path=args.output,
        intensity=args.intensity,
        preserve_geometry=args.preserve_geometry
    )

    print("✓ Veneer preview generated successfully!")