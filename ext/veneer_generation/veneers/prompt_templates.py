"""
Prompt Templates

Prompt engineering separate from geometry logic.
Production-ready prompts optimized for porcelain veneer generation.

CRITICAL: CLIP tokenizer has a hard limit of 77 tokens.
All prompts MUST stay under ~70 tokens to avoid silent truncation.
Weighted syntax (word:1.5) costs ~3-5 extra tokens per phrase — use sparingly.
Front-load the most important concepts (truncation cuts from the end).
"""

PROMPT_TEMPLATES = {
    "default": {
        "prompt": (
            # GEOMETRY LAYER (highest priority - front-loaded to survive truncation)
            "(complete smile reconstruction:1.8), "
            "(perfect symmetrical dental arch:1.7), "
            "(uniform tooth width and height:1.6), "
            "(ideal incisal edge alignment:1.5), "
            "(closed bite occlusion:1.5), "
            # MATERIAL LAYER
            "natural porcelain white, smooth enamel, subtle translucency, "
            # PHOTOGRAPHY LAYER (least critical - can be truncated)
            "professional dental photography, 85mm macro, photorealistic"
        ),
        "negative_prompt": (
            # AGGRESSIVE ANTI-GEOMETRY (front-loaded with high weights)
            "(original crooked teeth:1.7), "
            "(misaligned teeth:1.7), "
            "(gaps between teeth:1.7), "
            "(open bite:1.7), "
            "(asymmetrical smile:1.7), "
            # ANTI-MATERIAL
            "(yellow stains:1.6), "
            "(blue tint:1.6), "
            "(plastic look:1.6), "
            "(CGI render:1.6), "
            "(overexposed white:1.7), "
            "blurry teeth, cartoon, distorted gums"
        )
    },

    "natural": {
        "prompt": (
            # GEOMETRY LAYER (front-loaded)
            "(smile reconstruction with porcelain veneers:1.6), "
            "(symmetrical dental arch:1.5), "
            "(uniform tooth alignment:1.4), "
            "closed bite, even incisal edges, "
            # MATERIAL LAYER
            "natural white shade A1, smooth enamel, subtle translucency, "
            # PHOTOGRAPHY LAYER
            "dental photography, 85mm lens, photorealistic"
        ),
        "negative_prompt": (
            "(crooked teeth:1.6), "
            "(open bite:1.6), "
            "(gaps:1.6), "
            "(misaligned:1.6), "
            "overexposed, blue tint, plastic, yellow stains, "
            "CGI, cartoon, distorted gums"
        )
    },

    "dramatic": {
        "prompt": (
            # GEOMETRY LAYER (maximum weights for dramatic change)
            "(complete smile reconstruction:1.9), "
            "(flawless symmetrical dental arch:1.8), "
            "(perfect uniform tooth alignment:1.7), "
            "(ideal incisal edge plane:1.6), "
            "(closed bite occlusion:1.6), "
            # MATERIAL LAYER
            "brilliant porcelain white, smooth enamel, specular highlights, "
            # PHOTOGRAPHY LAYER
            "dental photography, 85mm macro, photorealistic"
        ),
        "negative_prompt": (
            "(original crooked teeth:1.8), "
            "(open bite:1.8), "
            "(gaps between teeth:1.8), "
            "(misaligned arch:1.8), "
            "(asymmetrical smile:1.8), "
            "overexposed, blue tint, plastic, CGI, cartoon, "
            "yellow stains, distorted gums"
        )
    },

    "conservative": {
        "prompt": (
            # GEOMETRY LAYER (moderate weights for subtle change)
            "(subtle veneer enhancement:1.3), "
            "(improved dental symmetry:1.2), "
            "closed bite, soft alignment correction, "
            # MATERIAL LAYER
            "natural white A2 shade, smooth enamel, subtle translucency, "
            # PHOTOGRAPHY LAYER
            "dental photography, soft lighting, photorealistic"
        ),
        "negative_prompt": (
            "(open bite:1.5), "
            "(severe misalignment:1.5), "
            "(large gaps:1.5), "
            "overexposed, plastic, blue tint, yellow stains, "
            "CGI, cartoon"
        )
    },
}


def get_prompts(template_name="default", custom_prompt=None, custom_negative_prompt=None):
    """
    Get prompt and negative prompt for generation.

    Args:
        template_name: Name of prompt template
        custom_prompt: Custom prompt override (optional)
        custom_negative_prompt: Custom negative prompt override (optional)

    Returns:
        tuple: (prompt, negative_prompt)
    """
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["default"])

    prompt = custom_prompt if custom_prompt is not None else template["prompt"]
    negative_prompt = custom_negative_prompt if custom_negative_prompt is not None else template["negative_prompt"]

    return prompt, negative_prompt


def list_templates():
    """
    List available prompt templates.

    Returns:
        list: Template names
    """
    return list(PROMPT_TEMPLATES.keys())