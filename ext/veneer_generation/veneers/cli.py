"""
CLI Interface for Veneer Generation

Clean command-line interface using the new modular architecture.
"""

import argparse
from pathlib import Path
from .pipeline import VeneerPipeline
from .presets import PRESETS
from .prompt_templates import list_templates


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate veneer preview using modular pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available presets: {', '.join(PRESETS.keys())}
Available prompt templates: {', '.join(list_templates())}

Examples:
  # Basic usage with natural preset
  python -m veneers.cli --controlnet ./weights/controlnet --image input.jpg --output output.jpg

  # Dramatic transformation
  python -m veneers.cli --controlnet ./weights/controlnet --image input.jpg --output output.jpg --preset dramatic

  # Custom parameters
  python -m veneers.cli --controlnet ./weights/controlnet --image input.jpg --output output.jpg --preset veneers_max --seed 42

  # With segmentation model
  python -m veneers.cli --controlnet ./weights/controlnet --segmentation ./weights/seg.pth --image input.jpg --output output.jpg

  # Side-by-side comparison
  python -m veneers.cli --controlnet ./weights/controlnet --image input.jpg --output comparison.jpg --comparison
"""
    )

    # Required arguments
    parser.add_argument(
        "--controlnet",
        type=str,
        required=True,
        help="Path to trained ControlNet model"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input smile image"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for veneer preview"
    )

    # Model configuration
    parser.add_argument(
        "--base-model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model (default: runwayml/stable-diffusion-v1-5)"
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        default=None,
        help="Path to segmentation model checkpoint (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )

    # Generation parameters
    parser.add_argument(
        "--preset",
        type=str,
        default="natural",
        choices=list(PRESETS.keys()),
        help="Parameter preset (default: natural)"
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="default",
        choices=list_templates(),
        help="Prompt template (default: default)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom text prompt (overrides template)"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Custom negative prompt (overrides template)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    # Debug options
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Directory to save debug images"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate side-by-side comparison"
    )

    args = parser.parse_args()

    # Initialize pipeline
    print(f"Initializing pipeline with preset: {args.preset}")
    pipeline = VeneerPipeline(
        controlnet_path=args.controlnet,
        base_model=args.base_model,
        segmentation_checkpoint=args.segmentation,
        device=args.device
    )

    if args.comparison:
        # Generate comparison
        result = pipeline.generate_comparison(
            image=args.image,
            output_path=args.output,
            preset=args.preset,
            prompt_template=args.prompt_template,
            custom_prompt=args.prompt,
            custom_negative_prompt=args.negative_prompt,
            seed=args.seed,
            debug_dir=args.debug_dir
        )
        print(f"✓ Comparison saved to {args.output}")
    else:
        # Generate single preview
        result = pipeline.run(
            image=args.image,
            preset=args.preset,
            prompt_template=args.prompt_template,
            custom_prompt=args.prompt,
            custom_negative_prompt=args.negative_prompt,
            seed=args.seed,
            debug_dir=args.debug_dir
        )
        result.save(args.output, quality=95)
        print(f"✓ Veneer preview saved to {args.output}")


if __name__ == "__main__":
    main()
