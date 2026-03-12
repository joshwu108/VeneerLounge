"""
Diffusion Engine

Handles ONLY Stable Diffusion + ControlNet inference.
No segmentation, no whitening, no post-processing.
"""

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, DPMSolverMultistepScheduler


class DiffusionEngine:
    """
    Manages Stable Diffusion + ControlNet pipeline.
    Single responsibility: generate images from conditioning.
    """

    def __init__(self, controlnet_path, base_model="runwayml/stable-diffusion-v1-5", device="cuda"):
        """
        Initialize diffusion pipeline.

        Args:
            controlnet_path: Path to trained ControlNet weights
            base_model: Base Stable Diffusion model path or HF model ID
            device: 'cuda' or 'cpu'
        """
        self.device = self._resolve_device(device)
        # float16 on MPS produces NaN/Inf in UNet output — force float32
        # CUDA is fine with float16 (no NaN issues)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print(f"Loading ControlNet from {controlnet_path}...")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=self.dtype
        )

        print(f"Loading Stable Diffusion base model {base_model}...")
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None
        )

        # DPM++ SDE Karras: better detail retention at low step counts
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True
        )

        self.pipe = self.pipe.to(self.device)

        # Performance optimizations
        if self.device.type in ("cuda", "mps"):
            self._optimize_for_gpu()

        print(f"✓ DiffusionEngine initialized on {self.device}")

    @staticmethod
    def _resolve_device(requested):
        """Resolve the best available device: cuda > mps > cpu."""
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _optimize_for_gpu(self):
        """Apply GPU-specific optimizations (CUDA and MPS)."""
        if self.device.type == "cuda":
            # Try xformers first (~25% speedup)
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✓ xformers enabled")
            except Exception:
                self.pipe.enable_attention_slicing(slice_size="auto")
                print("✓ attention slicing enabled")
        elif self.device.type == "mps":
            # MPS doesn't support xformers; use attention slicing
            self.pipe.enable_attention_slicing(slice_size="auto")
            print("✓ attention slicing enabled (MPS)")

        # VAE slicing for memory efficiency
        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass

        # torch.compile for CUDA only (not yet stable on MPS)
        if self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                self.pipe.controlnet = torch.compile(self.pipe.controlnet, mode="reduce-overhead")
                print("✓ torch.compile enabled")
            except Exception as e:
                print(f"torch.compile not available: {e}")

    def generate(
        self,
        image,
        mask,
        conditioning,
        prompt,
        negative_prompt,
        num_inference_steps=28,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.65,
        strength=0.88,
        seed=None
    ):
        """
        Generate veneer preview using ControlNet inpainting.

        Args:
            image: PIL Image (input photo)
            mask: PIL Image (teeth mask, grayscale)
            conditioning: PIL Image (multi-channel conditioning: mask+edges+distance)
            prompt: Text prompt
            negative_prompt: Negative text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            controlnet_conditioning_scale: ControlNet influence strength
            strength: Denoising strength (0-1, higher = more change)
            seed: Random seed for reproducibility

        Returns:
            PIL Image: Generated result
        """
        generator = None
        if seed is not None:
            # MPS requires CPU generator; CUDA uses its own device
            gen_device = "cpu" if self.device.type == "mps" else self.device
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Clear GPU cache before generation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            control_image=conditioning,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            generator=generator
        )

        # Clear GPU cache after generation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        return output.images[0]
