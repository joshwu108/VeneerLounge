"""
TEMPLATE 2: SDXL Two-Pass Geometry + Texture

Separates reconstruction from polish for better results.

Pass 1: High strength geometry rebuild (fixes alignment, shape)
Pass 2: Lower strength texture refinement (adds realism, texture)

Key features:
- Two-stage generation
- More stable than single-pass
- Better control over geometry vs texture
- No ControlNet
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import sys

from diffusers import StableDiffusionXLInpaintPipeline

# Import segmentation model
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'individual_tooth_segmentation'))
try:
    from src.network.model import ResNeSt50_TC as TeethSegmentationNet
except ImportError:
    TeethSegmentationNet = None


class VeneerSDXLTwoPassGenerator:
    """
    Two-pass SDXL veneer generator.

    Pass 1 (Geometry): High strength, focuses on tooth alignment and shape
    Pass 2 (Texture): Lower strength, refines surface texture and realism
    """

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        segmentation_checkpoint=None,
        device="cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        print("Loading SDXL Inpainting pipeline for two-pass generation...")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(self.device)

        if self.device.type == 'cuda':
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✓ xformers enabled")
            except Exception:
                pass
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

        print("✓ VeneerSDXLTwoPassGenerator initialized")

    def generate_tooth_mask(self, image, target_size=None):
        """Generate tooth segmentation mask."""
        if target_size:
            image_resized = image.resize(target_size, Image.LANCZOS)
        else:
            image_resized = image
            target_size = image.size

        if self.seg_model is None:
            img_array = np.array(image_resized)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            return Image.fromarray(mask, mode='L')

        img_array = np.array(image_resized)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            output = self.seg_model(img_tensor)
            mask = torch.sigmoid(output) > 0.5
            mask = mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        return Image.fromarray(mask, mode='L')

    def refine_mask(self, mask, erosion_px=4, feather_px=15):
        """Refine mask with erosion and feathering."""
        mask_np = np.array(mask)

        if erosion_px > 0:
            kernel = np.ones((erosion_px, erosion_px), np.uint8)
            mask_np = cv2.erode(mask_np, kernel, iterations=1)

        if feather_px > 0:
            mask_np = cv2.GaussianBlur(mask_np, (feather_px * 2 + 1, feather_px * 2 + 1), 0)

        return Image.fromarray(mask_np, mode='L')

    def generate_pass1_geometry(self, image, mask, generator=None):
        """
        Pass 1: Geometry reconstruction.

        High strength, focuses on:
        - Tooth alignment
        - Shape correction
        - Symmetry
        """
        prompt = """
        perfectly aligned symmetrical porcelain veneers,
        ideal dental arch curvature,
        uniform incisal edge line,
        correct tooth proportions,
        professional cosmetic dentistry result
        """

        negative_prompt = """
        distorted face, plastic teeth, crooked teeth,
        uneven teeth, gaps, misaligned, asymmetric
        """

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=0.90,  # High strength for geometry changes
            guidance_scale=9.5,
            num_inference_steps=40,
            generator=generator
        )

        return output.images[0]

    def generate_pass2_texture(self, image, mask, generator=None):
        """
        Pass 2: Texture refinement.

        Lower strength, focuses on:
        - Natural enamel texture
        - Subtle translucency
        - Realistic surface details
        """
        prompt = """
        natural enamel microtexture,
        smooth porcelain surface,
        subtle incisal translucency,
        neutral white shade,
        balanced lighting,
        photorealistic dental photography
        """

        negative_prompt = """
        plastic teeth, blue tint, overexposed white,
        CGI, flat texture, matte finish
        """

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=0.55,  # Lower strength to preserve geometry
            guidance_scale=7.5,
            num_inference_steps=25,
            generator=generator
        )

        return output.images[0]

    def generate(
        self,
        image,
        mask=None,
        bounding_box=None,
        seed=None,
        debug_dir=None,
        skip_pass2=False
    ):
        """
        Generate veneer preview using two-pass approach.

        Args:
            image: PIL Image or path to image
            mask: Optional pre-computed mask
            bounding_box: Optional dict with x, y, width, height (percentages)
            seed: Random seed for reproducibility
            debug_dir: Directory to save debug images
            skip_pass2: If True, only run geometry pass

        Returns:
            PIL Image with veneer preview
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        original = image.copy()
        orig_w, orig_h = original.size

        # Determine crop region
        if bounding_box:
            x = int(bounding_box['x'] / 100.0 * orig_w)
            y = int(bounding_box['y'] / 100.0 * orig_h)
            w = int(bounding_box['width'] / 100.0 * orig_w)
            h = int(bounding_box['height'] / 100.0 * orig_h)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(orig_w, x + w), min(orig_h, y + h)
        else:
            x1, y1 = 0, int(orig_h * 0.5)
            x2, y2 = orig_w, orig_h

        crop = original.crop((x1, y1, x2, y2))
        crop_w, crop_h = crop.size

        # Maintain aspect ratio
        target_w = (crop_w // 8) * 8
        target_h = (crop_h // 8) * 8
        target_w = max(512, min(1024, target_w))
        target_h = max(512, min(1024, target_h))

        crop_resized = crop.resize((target_w, target_h), Image.LANCZOS)

        # Generate mask
        if mask is None:
            tooth_mask = self.generate_tooth_mask(crop_resized)
        else:
            tooth_mask = mask.resize((target_w, target_h), Image.NEAREST)

        tooth_mask = self.refine_mask(tooth_mask, erosion_px=4, feather_px=12)

        # Check mask
        mask_np = np.array(tooth_mask)
        if mask_np.mean() < 5:
            print("Warning: Empty mask, creating fallback")
            mask_np = np.zeros((target_h, target_w), dtype=np.uint8)
            h_start, h_end = int(target_h * 0.3), int(target_h * 0.7)
            w_start, w_end = int(target_w * 0.15), int(target_w * 0.85)
            mask_np[h_start:h_end, w_start:w_end] = 255
            tooth_mask = Image.fromarray(mask_np, mode='L')

        # Setup generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Debug output
        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            crop_resized.save(debug_path / "01_input_crop.png")
            tooth_mask.save(debug_path / "02_tooth_mask.png")

        # PASS 1: Geometry
        print("Running Pass 1: Geometry reconstruction...")
        pass1_result = self.generate_pass1_geometry(crop_resized, tooth_mask, generator)

        if debug_dir:
            pass1_result.save(debug_path / "03_pass1_geometry.png")

        # PASS 2: Texture (optional)
        if not skip_pass2:
            print("Running Pass 2: Texture refinement...")
            # Use same seed for consistency
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed + 1)
            pass2_result = self.generate_pass2_texture(pass1_result, tooth_mask, generator)
            generated = pass2_result

            if debug_dir:
                pass2_result.save(debug_path / "04_pass2_texture.png")
        else:
            generated = pass1_result

        # Resize and blend
        generated_resized = generated.resize((crop_w, crop_h), Image.LANCZOS)

        blend_mask = tooth_mask.resize((crop_w, crop_h), Image.LANCZOS)
        blend_mask_np = np.array(blend_mask).astype(np.float32) / 255.0

        result = np.array(original).astype(np.float32)
        generated_np = np.array(generated_resized).astype(np.float32)

        for c in range(3):
            result[y1:y2, x1:x2, c] = (
                generated_np[:, :, c] * blend_mask_np +
                result[y1:y2, x1:x2, c] * (1 - blend_mask_np)
            )

        result = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

        if debug_dir:
            result.save(debug_path / "05_final_output.png")
            print(f"✓ Debug images saved to {debug_dir}")

        return result


def main():
    """CLI interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description='SDXL Two-Pass Veneer Generator')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--segmentation', type=str, default=None, help='Segmentation model path')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--debug-dir', type=str, default=None, help='Debug output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--skip-pass2', action='store_true', help='Skip texture refinement pass')

    args = parser.parse_args()

    generator = VeneerSDXLTwoPassGenerator(
        segmentation_checkpoint=args.segmentation,
        device=args.device
    )

    result = generator.generate(
        image=args.image,
        seed=args.seed,
        debug_dir=args.debug_dir,
        skip_pass2=args.skip_pass2
    )

    result.save(args.output, quality=95)
    print(f"✓ Output saved to {args.output}")


if __name__ == "__main__":
    main()
