"""
TEMPLATE 1: SDXL Inpainting (No ControlNet, Full Face)

Best balance of realism + stability.
This is the recommended baseline approach.

Key features:
- Uses SDXL inpainting
- No ControlNet (removes distribution shift issues)
- No square cropping (preserves facial anchors)
- Full face context preserved
- Teeth mask only
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import sys

from diffusers import StableDiffusionXLInpaintPipeline

# Import segmentation model
_seg_model_path = Path(__file__).resolve().parent.parent.parent / 'individual_tooth_segmentation'
sys.path.insert(0, str(_seg_model_path))
try:
    from src.network.model import ResNeSt50_TC as TeethSegmentationNet
    print(f"✓ Segmentation module imported from {_seg_model_path}")
except ImportError as e:
    print(f"Warning: Could not import segmentation model: {e}")
    TeethSegmentationNet = None


class VeneerSDXLGenerator:
    """
    SDXL-based veneer generator without ControlNet.

    Advantages:
    - Higher quality output from SDXL
    - No distribution shift from custom conditioning
    - Full face context preserved (no square cropping)
    - More stable and predictable results
    """

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        segmentation_checkpoint=None,
        device="cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        print("Loading SDXL Inpainting pipeline...")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(self.device)

        # Enable memory optimizations
        if self.device.type == 'cuda':
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✓ xformers enabled")
            except Exception:
                print("xformers not available, using default attention")
            self.pipe.enable_model_cpu_offload()

        # Load segmentation model
        self.seg_model = None
        if segmentation_checkpoint and TeethSegmentationNet:
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
                self.seg_model = None

        print("✓ VeneerSDXLGenerator initialized")

    def generate_tooth_mask(self, image, target_size=(768, 768)):
        """Generate tooth segmentation mask."""
        image_resized = image.resize(target_size, Image.LANCZOS)

        if self.seg_model is None:
            # Fallback: color-based detection
            print("Warning: Using fallback color-based segmentation")
            img_array = np.array(image_resized)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            return Image.fromarray(mask, mode='L')

        print("Using neural segmentation model...")
        img_array = np.array(image_resized)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            output = self.seg_model(img_tensor)
            print(f"Segmentation output shape: {output.shape}, min: {output.min():.3f}, max: {output.max():.3f}")
            mask = torch.sigmoid(output) > 0.5
            mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            print(f"Mask sum: {mask.sum()}, mean: {mask.mean():.1f}")

        return Image.fromarray(mask, mode='L')

    def refine_mask(self, mask, erosion_px=4, feather_px=15):
        """Refine mask with erosion and feathering."""
        mask_np = np.array(mask)

        # Erode to ensure we stay within teeth
        if erosion_px > 0:
            kernel = np.ones((erosion_px, erosion_px), np.uint8)
            mask_np = cv2.erode(mask_np, kernel, iterations=1)

        # Feather edges for smooth blending
        if feather_px > 0:
            mask_np = cv2.GaussianBlur(mask_np, (feather_px * 2 + 1, feather_px * 2 + 1), 0)

        return Image.fromarray(mask_np, mode='L')

    def generate(
        self,
        image,
        mask=None,
        bounding_box=None,
        strength=0.85,
        guidance_scale=9.0,
        num_inference_steps=40,
        seed=None,
        debug_dir=None
    ):
        """
        Generate veneer preview.

        Args:
            image: PIL Image or path to image
            mask: Optional pre-computed mask (will generate if None)
            bounding_box: Optional dict with x, y, width, height (percentages)
            strength: Denoising strength (0.0-1.0)
            guidance_scale: CFG scale
            num_inference_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            debug_dir: Directory to save debug images

        Returns:
            PIL Image with veneer preview
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        original = image.copy()
        orig_w, orig_h = original.size

        # Determine crop region (full image or bounding box)
        if bounding_box:
            x = int(bounding_box['x'] / 100.0 * orig_w)
            y = int(bounding_box['y'] / 100.0 * orig_h)
            w = int(bounding_box['width'] / 100.0 * orig_w)
            h = int(bounding_box['height'] / 100.0 * orig_h)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(orig_w, x + w), min(orig_h, y + h)
        else:
            # Use lower 50% of image by default (where mouth typically is)
            x1, y1 = 0, int(orig_h * 0.5)
            x2, y2 = orig_w, orig_h

        # Crop to region of interest
        crop = original.crop((x1, y1, x2, y2))
        crop_w, crop_h = crop.size

        # IMPORTANT: Maintain aspect ratio, don't force square
        # SDXL works with various sizes, but prefers multiples of 8
        target_w = (crop_w // 8) * 8
        target_h = (crop_h // 8) * 8
        target_w = max(512, min(1024, target_w))
        target_h = max(512, min(1024, target_h))

        crop_resized = crop.resize((target_w, target_h), Image.LANCZOS)

        # Generate or resize mask
        if mask is None:
            tooth_mask = self.generate_tooth_mask(crop_resized)
        else:
            tooth_mask = mask.resize((target_w, target_h), Image.NEAREST)

        # Refine mask
        tooth_mask = self.refine_mask(tooth_mask, erosion_px=4, feather_px=12)

        # Check if mask has content
        mask_np = np.array(tooth_mask)
        if mask_np.mean() < 5:
            print("Warning: Empty mask, creating fallback")
            mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
            h_start, h_end = int(target_h * 0.3), int(target_h * 0.7)
            w_start, w_end = int(target_w * 0.15), int(target_w * 0.85)
            mask_np[h_start:h_end, w_start:w_end] = 255
            tooth_mask = Image.fromarray(mask_np, mode='L')

        # Setup generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Prompts optimized for dental work
        prompt = """
        complete cosmetic smile reconstruction with ideal porcelain veneers,
        perfectly symmetrical teeth,
        uniform incisal edge alignment,
        natural enamel translucency,
        professional dental photography,
        photorealistic
        """

        negative_prompt = """
        distorted face, changed identity, altered lips, altered skin,
        plastic teeth, CGI, cartoon, blurry, unnatural gloss,
        crooked teeth, yellow teeth, gaps between teeth
        """

        # Save debug images
        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            crop_resized.save(debug_path / "01_input_crop.png")
            tooth_mask.save(debug_path / "02_tooth_mask.png")

        # Run SDXL inpainting
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=crop_resized,
            mask_image=tooth_mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )

        generated = output.images[0]

        if debug_dir:
            generated.save(debug_path / "03_generated.png")

        # Resize back to crop dimensions
        generated_resized = generated.resize((crop_w, crop_h), Image.LANCZOS)

        # Prepare blending mask
        blend_mask = tooth_mask.resize((crop_w, crop_h), Image.LANCZOS)
        blend_mask_np = np.array(blend_mask).astype(np.float32) / 255.0

        # Debug dimensions
        print(f"DEBUG: original size: {original.size}")
        print(f"DEBUG: crop region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"DEBUG: crop_w={crop_w}, crop_h={crop_h}")
        print(f"DEBUG: generated_resized size: {generated_resized.size}")
        print(f"DEBUG: blend_mask_np shape: {blend_mask_np.shape}")
        print(f"DEBUG: blend_mask_np min={blend_mask_np.min():.3f}, max={blend_mask_np.max():.3f}")

        # Alpha blend back into original
        result = np.array(original).astype(np.float32)
        generated_np = np.array(generated_resized).astype(np.float32)

        print(f"DEBUG: result shape: {result.shape}")
        print(f"DEBUG: generated_np shape: {generated_np.shape}")
        print(f"DEBUG: target slice shape: {result[y1:y2, x1:x2, 0].shape}")

        for c in range(3):
            result[y1:y2, x1:x2, c] = (
                generated_np[:, :, c] * blend_mask_np +
                result[y1:y2, x1:x2, c] * (1 - blend_mask_np)
            )

        result = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

        if debug_dir:
            result.save(debug_path / "04_final_output.png")
            print(f"✓ Debug images saved to {debug_dir}")

        return result


def main():
    """CLI interface for testing."""
    import argparse
    from PIL import Image as PILImage

    parser = argparse.ArgumentParser(description='SDXL Veneer Generator (No ControlNet)')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--segmentation', type=str, default=None, help='Segmentation model path')
    parser.add_argument('--mask', type=str, default=None, help='Manual mask image path (overrides segmentation)')
    parser.add_argument('--strength', type=float, default=0.85, help='Denoising strength')
    parser.add_argument('--guidance', type=float, default=9.0, help='Guidance scale')
    parser.add_argument('--steps', type=int, default=40, help='Inference steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--debug-dir', type=str, default=None, help='Debug output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    generator = VeneerSDXLGenerator(
        segmentation_checkpoint=args.segmentation,
        device=args.device
    )

    # Load manual mask if provided
    manual_mask = None
    if args.mask:
        manual_mask = PILImage.open(args.mask).convert('L')
        print(f"Using manual mask from {args.mask}")

    result = generator.generate(
        image=args.image,
        mask=manual_mask,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
        debug_dir=args.debug_dir
    )

    result.save(args.output, quality=95)
    print(f"✓ Output saved to {args.output}")


if __name__ == "__main__":
    main()
