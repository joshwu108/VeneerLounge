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
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'individual_tooth_segmentation'))
from src.network.model import ResNeSt50_TC as TeethSegmentationNet

logger = logging.getLogger(__name__)

class VeneerControlNetGenerator:

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
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        if self.device.type == 'cuda':
            self.pipe.enable_model_cpu_offload()

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
        Generate edge map using Canny detection.
        Args:
            image: PIL Image
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold

        Returns:
            PIL Image of edge map
        """
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

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
        mask_Np = cv2.GaussianBlur(mask_np, (9, 9), 0)
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

    def refine_tooth_mask(self, mask, erosion_px=6):
        mask_np = np.array(mask)
        kernel = np.ones((erosion_px, erosion_px), np.uint8)
        refined = cv2.erode(mask_np, kernel, iterations=1)
        return Image.fromarray(refined, mode='L')

    def advanced_feather_mask(self, mask, inner_feather_px=10, outer_feather_px=35):
        """
        Two-stage feathering: tight near teeth, wide at boundaries.
        Better preserves tooth detail while ensuring smooth blending.
        """
        mask_np = (np.array(mask) > 0).astype(np.uint8) * 255
        dist_inner = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
        inner_feather = np.clip(dist_inner / inner_feather_px, 0, 1)
        inverted = 255 - mask_np
        dist_outer = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
        outer_feather = 1 - np.clip(dist_outer / outer_feather_px, 0, 1)
        combined = np.minimum(inner_feather, outer_feather)

        return (combined * 255).astype(np.uint8)

    def poisson_blend(self, generated_crop, original, mask, offset):
        """Seamless cloning using Poisson editing."""
        x1, y1 = offset
        x2, y2 = x1 + generated_crop.width, y1 + generated_crop.height
        mask_np = np.array(mask)
        _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        center = (x1 + generated_crop.width // 2, y1 + generated_crop.height // 2)
        original_np = np.array(original)
        generated_np = np.array(generated_crop)

        result = cv2.seamlessClone(
            generated_np,
            original_np,
            binary_mask,
            center,
            cv2.NORMAL_CLONE
        )

        return Image.fromarray(result)

    def alpha_blend(self, generated_crop, original, mask, offset):
        """
        Simple alpha blending - reliable and artifact-free.
        """
        x1, y1 = offset
        original_np = np.array(original).astype(np.float32)
        generated_np = np.array(generated_crop).astype(np.float32)
        mask_np = np.array(mask).astype(np.float32) / 255.0

        # Ensure mask is 3-channel
        if len(mask_np.shape) == 2:
            mask_3ch = np.stack([mask_np] * 3, axis=-1)
        else:
            mask_3ch = mask_np

        # Extract region from original
        h, w = generated_np.shape[:2]
        orig_region = original_np[y1:y1+h, x1:x1+w].copy()

        # Simple alpha blend: result = gen * mask + orig * (1 - mask)
        blended = generated_np * mask_3ch + orig_region * (1.0 - mask_3ch)

        # Place back
        output = original_np.copy()
        output[y1:y1+h, x1:x1+w] = blended

        return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


    def generate_veneer_preview(
        self,
        image,
        prompt=None,
        negative_prompt=None,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.45,
        strength=0.80,
        seed=None,
        debug_dir=None,
        bounding_box=None,
        use_two_pass=False
    ):
        """
        Generate veneer preview for an input smile image.

        Args:
            image: PIL Image or path to image
            prompt: Text prompt (optional, uses default if None)
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps (default: 30)
            use_two_pass: Whether to use two-pass generation (slower but higher quality)
            guidance_scale: Classifier-free guidance scale (default: 7.5)
            controlnet_conditioning_scale: How much to follow conditioning (default: 0.45)
            strength: Denoising strength (default: 0.80)
            seed: Random seed for reproducibility
            debug_dir: Directory to save debug images (optional)
            bounding_box: bounding box for mouth region
        Returns:
            PIL Image of veneer preview
        """
        # Load image if path
        GEN_SIZE = 768
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        output_image = image.copy()
        orig_w, orig_h = output_image.size

        #default use bottom 40% of image
        x1, y1, x2, y2 = 0, int(orig_h * 0.6), int(orig_w), int(orig_h)

        if bounding_box:
            #extracting bounding box coordinates
            x, y = bounding_box['x'], bounding_box['y']
            width, height = bounding_box['width'], bounding_box['height']

            x1, y1 = int(x/100.0 * orig_w), int(y/100.0 * orig_h)
            x2, y2 = int((x + width)/100.0 * orig_w), int((y + height)/100.0 * orig_h)
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

        # crop to bounding box
        image = image.crop((x1, y1, x2, y2))
        orig_w, orig_h = image.size
        image = image.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)

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
            # Conservative erosion - only teeth, preserve full tooth area
            tooth_mask = self.refine_tooth_mask(tooth_mask, erosion_px=4)

        mask_np = np.array(tooth_mask)
        h, w = mask_np.shape

        # Tight feathering for clean teeth-only modification
        mask_np = self.advanced_feather_mask(tooth_mask, inner_feather_px=5, outer_feather_px=15)
        tooth_mask = Image.fromarray(mask_np, mode="L")

        if debug_dir:
            tooth_mask.save(Path(debug_dir) / "gated_tooth_mask.png")
            print(f"Debug: Mask stats - min={mask_np.min()}, max={mask_np.max()}, mean={mask_np.mean():.1f}")

        offset = (x1, y1)
        crop_img, crop_mask = image, tooth_mask

        sd_crop_img = crop_img.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)
        sd_crop_mask = crop_mask.resize((GEN_SIZE, GEN_SIZE), Image.NEAREST)

        mask_np = np.array(sd_crop_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        #mask_np = cv2.erode(mask_np, kernel, iterations=1)

        mask_bin = (mask_np > 0).astype(np.uint8)

        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
        dist_norm = np.clip(dist / 16.0, 0, 1) * 255
        dist_norm = dist_norm.astype(np.uint8)

        edges = self.generate_edge_map(sd_crop_img)
        edges_np = np.array(edges)
        edges_np = edges_np * mask_bin

        defect_region = (dist < 5) & (mask_bin == 1)
        edges_np[defect_region] = 0

        conditioning = np.stack(
            [mask_np, edges_np, dist_norm],
            axis=-1
        )

        conditioning_image = Image.fromarray(conditioning)

        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            tooth_mask.save(debug_path / 'tooth_mask.png')
            image.save(debug_path / 'input_resized.png')
            conditioning_image.save(debug_path / 'conditioning_edges.png')

        if prompt is None:
            prompt = """perfect white dental veneers, bright uniform teeth,
                        perfectly aligned straight teeth, natural enamel texture,
                        professional teeth whitening, pristine dental work,
                        flawless smile, perfectly symmetrical teeth, photorealistic"""
        if negative_prompt is None:
            negative_prompt = """yellow teeth, stained teeth, crooked teeth, misaligned teeth,
                                plastic teeth, fake smile, porcelain doll, sharp edges,
                                distorted lips, altered gums, altered face, uncanny, cartoon, CGI,
                                gaps between teeth, uneven teeth"""
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # ---------------------------
        # PASS 1: GEOMETRY REBUILD
        # ---------------------------

        output = self.pipe(
            prompt="""
            perfect white dental veneers only, natural tooth texture,
            sharp focus on teeth, pristine enamel,
            professional cosmetic dentistry, high resolution teeth detail,
            maintain original face and skin unchanged,
            photorealistic dental work, crisp tooth edges
            """,
            negative_prompt="""
            blurry, distorted face, changed face, altered skin, modified lips,
            face modification, facial distortion, soft focus, low quality,
            crooked teeth, yellow teeth, plastic teeth, fake teeth, CGI teeth,
            cartoon, painting, illustration
            """,
            image=sd_crop_img,
            mask_image=sd_crop_mask,
            control_image=conditioning_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            generator=generator
        )

        generated_sd = output.images[0]

        # ---------------------------
        # PASS 2: TEXTURE + POLISH (Optional)
        # ---------------------------
        if use_two_pass:
            output2 = self.pipe(
                prompt="""
                natural porcelain veneer texture,
                smooth enamel microtexture,
                subtle incisal translucency,
                neutral white shade,
                balanced lighting,
                photorealistic dental photography
                """,
                negative_prompt="plastic teeth, blue tint, overexposed white, CGI, flat texture",
                image=generated_sd,
                mask_image=sd_crop_mask,
                control_image=conditioning_image,
                num_inference_steps=15,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.25,
                strength=0.55,
                generator=generator
            )
            generated_sd = output2.images[0]

        # Resize generated image back to the original bounding box dimensions
        bbox_w, bbox_h = x2 - x1, y2 - y1
        generated_crop = generated_sd.resize((bbox_w, bbox_h), Image.LANCZOS)

        # Prepare final mask for alpha blending
        final_mask = sd_crop_mask.resize((bbox_w, bbox_h), Image.LANCZOS)
        # Heavy blur for smooth seamless blend
        mask_np = np.array(final_mask)
        mask_np = cv2.GaussianBlur(mask_np, (51, 51), 0)
        final_mask = Image.fromarray(mask_np, mode="L")

        # Simple alpha blending - no face warping
        output_image = self.alpha_blend(generated_crop, output_image, final_mask, (x1, y1))

        if debug_dir:
            output_image.save(Path(debug_dir) / 'output.png', compress_level=0)

        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            sd_crop_img.save(debug_path / "01_input.png")
            sd_crop_mask.save(debug_path / "02_mask.png")
            conditioning_image.save(debug_path / "03_conditioning.png")
            generated_sd.save(debug_path / "04_pass2_output.png")

            delta = np.mean(
                np.abs(np.array(sd_crop_img).astype(np.int32) -
                        np.array(generated_sd).astype(np.int32))
            )

            print(f"Diffusion delta score: {delta:.2f}")

            if delta < 5:
                print("⚠ WARNING: Diffusion ineffective — increase strength or lower ControlNet scale")

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
        GEN_SIZE = 768
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
        default=30,
        help='Number of inference steps'
    )
    parser.add_argument(
        '--guidance',
        type=float,
        default=7.5,
        help='Guidance scale'
    )
    parser.add_argument(
        '--controlnet-scale',
        type=float,
        default=0.45,
        help='ControlNet conditioning scale'
    )
    parser.add_argument(
        '--strength',
        type=float,
        default=0.80,
        help='Denoising strength'
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