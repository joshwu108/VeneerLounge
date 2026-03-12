"""
TEMPLATE 4: SDXL + ControlNet (Properly Trained)

Only use this if your ControlNet was trained on:
- Pure Canny edges, OR
- Pure segmentation masks

Do NOT stack custom RGB conditioning unless the model was trained that way.

Key settings:
- controlnet_conditioning_scale = 0.25 (LOW - avoid over-constraining)
- strength = 0.85
- guidance_scale = 9

WARNING: Using this with an improperly trained ControlNet will cause
the same issues you've been experiencing.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import sys

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    UniPCMultistepScheduler
)

# Import segmentation model
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'individual_tooth_segmentation'))
try:
    from src.network.model import ResNeSt50_TC as TeethSegmentationNet
except ImportError:
    TeethSegmentationNet = None


class VeneerSDXLControlNetGenerator:
    """
    SDXL + ControlNet generator.

    IMPORTANT: Only use if your ControlNet model was properly trained on
    the same type of conditioning you're providing (Canny OR segmentation,
    not both stacked).

    Critical settings to avoid over-constraining:
    - controlnet_conditioning_scale: 0.25 (default, very low)
    - strength: 0.85
    - guidance_scale: 9.0
    """

    def __init__(
        self,
        controlnet_path,
        base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        segmentation_checkpoint=None,
        conditioning_type="canny",  # "canny" or "segmentation"
        device="cuda"
    ):
        """
        Initialize the generator.

        Args:
            controlnet_path: Path to trained ControlNet weights
            base_model_id: SDXL base model ID
            segmentation_checkpoint: Path to tooth segmentation model
            conditioning_type: Type of conditioning ("canny" or "segmentation")
            device: cuda or cpu
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conditioning_type = conditioning_type
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        print(f"Loading ControlNet from {controlnet_path}...")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=dtype
        )

        print("Loading SDXL + ControlNet Inpainting pipeline...")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        ).to(self.device)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

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

        print(f"✓ VeneerSDXLControlNetGenerator initialized (conditioning: {conditioning_type})")

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

    def generate_canny_edges(self, image, low_threshold=100, high_threshold=200):
        """Generate Canny edge map for conditioning."""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert to 3-channel for ControlNet
        edges_3ch = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges_3ch)

    def generate_segmentation_conditioning(self, mask):
        """Generate segmentation-based conditioning image."""
        mask_np = np.array(mask)
        # Convert to 3-channel
        conditioning = np.stack([mask_np] * 3, axis=-1)
        return Image.fromarray(conditioning)

    def generate_conditioning_image(self, image, mask):
        """
        Generate appropriate conditioning image based on conditioning_type.

        IMPORTANT: Only use ONE type of conditioning - do NOT stack multiple
        conditioning types unless the model was trained that way.
        """
        if self.conditioning_type == "canny":
            return self.generate_canny_edges(image)
        elif self.conditioning_type == "segmentation":
            return self.generate_segmentation_conditioning(mask)
        else:
            raise ValueError(f"Unknown conditioning type: {self.conditioning_type}")

    def refine_mask(self, mask, erosion_px=4, feather_px=15):
        """Refine mask with erosion and feathering."""
        mask_np = np.array(mask)

        if erosion_px > 0:
            kernel = np.ones((erosion_px, erosion_px), np.uint8)
            mask_np = cv2.erode(mask_np, kernel, iterations=1)

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
        controlnet_conditioning_scale=0.25,  # CRITICAL: Keep this LOW
        num_inference_steps=40,
        seed=None,
        debug_dir=None
    ):
        """
        Generate veneer preview using SDXL + ControlNet.

        Args:
            image: PIL Image or path to image
            mask: Optional pre-computed mask
            bounding_box: Optional dict with x, y, width, height (percentages)
            strength: Denoising strength (0.0-1.0)
            guidance_scale: CFG scale
            controlnet_conditioning_scale: ControlNet influence (keep LOW: 0.2-0.3)
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

        # Generate conditioning image (single type only!)
        conditioning_image = self.generate_conditioning_image(crop_resized, tooth_mask)

        # Setup generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Prompts
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

        # Debug output
        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            crop_resized.save(debug_path / "01_input_crop.png")
            tooth_mask.save(debug_path / "02_tooth_mask.png")
            conditioning_image.save(debug_path / "03_conditioning.png")

        # Run SDXL + ControlNet inpainting
        # CRITICAL: Keep controlnet_conditioning_scale LOW (0.2-0.3)
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=crop_resized,
            mask_image=tooth_mask,
            control_image=conditioning_image,
            strength=strength,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )

        generated = output.images[0]

        if debug_dir:
            generated.save(debug_path / "04_generated.png")

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

    parser = argparse.ArgumentParser(description='SDXL + ControlNet Veneer Generator')
    parser.add_argument('--controlnet', type=str, required=True, help='ControlNet model path')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--segmentation', type=str, default=None, help='Segmentation model path')
    parser.add_argument('--conditioning-type', type=str, default='canny',
                        choices=['canny', 'segmentation'], help='Type of ControlNet conditioning')
    parser.add_argument('--strength', type=float, default=0.85, help='Denoising strength')
    parser.add_argument('--guidance', type=float, default=9.0, help='Guidance scale')
    parser.add_argument('--controlnet-scale', type=float, default=0.25,
                        help='ControlNet conditioning scale (keep LOW: 0.2-0.3)')
    parser.add_argument('--steps', type=int, default=40, help='Inference steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--debug-dir', type=str, default=None, help='Debug output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    generator = VeneerSDXLControlNetGenerator(
        controlnet_path=args.controlnet,
        segmentation_checkpoint=args.segmentation,
        conditioning_type=args.conditioning_type,
        device=args.device
    )

    result = generator.generate(
        image=args.image,
        strength=args.strength,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.controlnet_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
        debug_dir=args.debug_dir
    )

    result.save(args.output, quality=95)
    print(f"✓ Output saved to {args.output}")


if __name__ == "__main__":
    main()
