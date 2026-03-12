"""
Unified Veneer Preview Service

This service provides a single interface for veneer preview generation with
- ControlNet (recommended for geometric + color changes)
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
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'ext' / 'veneer_generation'))


class VeneerPreviewService:
    """
    Unified service for veneer preview generation.
    Supports multiple model backends with a consistent interface.
    """

    def __init__(self, model_type='pix2pix', **model_config):
        """
        Initialize the veneer preview service.

        Args:
            model_type: Type of model to use ('controlnet' or 'pix2pix')
            **model_config: Model-specific configuration
                For ControlNet:
                    - controlnet_path: Path to ControlNet weights
                    - base_model_path: Path to Stable Diffusion base
                    - segmentation_checkpoint: Path to segmentation model
                For Pix2pix:
                    - checkpoint_path: Path to trained pix2pix model
                    - segmentation_checkpoint: Path to segmentation model
        """
        self.model_type = model_type
        self.generator = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing Veneer Preview Service...")
        print(f"  Model type: {model_type}")
        print(f"  Device: {self.device}")

        # Initialize the appropriate model
        self._initialize_generator(model_config)

    def _initialize_generator(self, config):
        """
        Initialize the veneer generator based on model type.

        Args:
            config: Model-specific configuration dictionary
        """
        if self.model_type == 'controlnet':
            self._init_controlnet(config)
        elif self.model_type == 'pix2pix':
            self._init_pix2pix(config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _init_controlnet(self, config):
        """Initialize ControlNet generator using new modular architecture."""
        from veneers import VeneerPipeline

        # Only controlnet_path is required, segmentation is optional
        if 'controlnet_path' not in config:
            raise ValueError(f"Missing required config key for ControlNet: controlnet_path")

        self.generator = VeneerPipeline(
            controlnet_path=config['controlnet_path'],
            base_model=config.get('base_model_path', 'runwayml/stable-diffusion-v1-5'),
            segmentation_checkpoint=config.get('segmentation_checkpoint', None),
            device=str(self.device)
        )

        print("✓ ControlNet pipeline initialized")

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
        if self.model_type == 'controlnet':
            return self._generate_controlnet(
                image, intensity, preserve_geometry, custom_prompt, bounding_box, **kwargs
            )
        elif self.model_type == 'pix2pix':
            return self._generate_pix2pix(image, intensity, **kwargs)

    def _generate_controlnet(
        self,
        image,
        intensity,
        preserve_geometry,
        custom_prompt,
        bounding_box,
        **kwargs
    ):
        # Use preset-based parameters from new modular architecture
        from veneers import get_preset, interpolate_presets

        if preserve_geometry:
            preset_name = 'conservative'
        else:
            # Map intensity (0-1) to progressively stronger presets
            t = max(0.0, min(1.0, intensity if intensity is not None else 0.8))
            if t < 0.3:
                preset_name = 'conservative'
            elif t < 0.6:
                preset_name = 'natural'
            elif t < 0.8:
                preset_name = 'balanced'
            else:
                preset_name = 'dramatic'

        # Enable debug directory
        import os
        debug_dir = Path(__file__).parent.parent.parent / 'debug_outputs'
        os.makedirs(debug_dir, exist_ok=True)

        # Use matching prompt template when available, else fall back to default
        template_name = preset_name if preset_name in ('natural', 'dramatic', 'conservative') else 'default'

        # Run the new pipeline
        result = self.generator.run(
            image=image,
            preset=preset_name,
            prompt_template=template_name,
            custom_prompt=custom_prompt,
            custom_negative_prompt=None,
            seed=kwargs.get('seed', None),
            debug_dir=str(debug_dir),
            bounding_box=bounding_box,
            two_pass=kwargs.get('two_pass', False)
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
            preview.save(buffer, format='JPEG', quality=95)
            preview_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return f"data:image/jpeg;base64,{preview_base64}"

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


def get_veneer_service(model_type='controlnet', force_reload=False, **config):
    """
    Get singleton instance of veneer service.

    Args:
        model_type: Type of model ('controlnet' or 'pix2pix')
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
    parser.add_argument('--model', type=str, default='controlnet', choices=['controlnet', 'pix2pix'])
    parser.add_argument('--controlnet-path', type=str, help='Path to ControlNet weights')
    parser.add_argument('--segmentation', type=str, help='Path to segmentation checkpoint')
    parser.add_argument('--intensity', type=float, default=0.8, help='Transformation intensity')
    parser.add_argument('--preserve-geometry', action='store_true', help='Preserve tooth positions')

    args = parser.parse_args()

    # Configure service
    if args.model == 'controlnet':
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