# stable_diffusion_generator.py
"""Generate fantasy scenes with Stable Diffusion – now with **command-line arguments**
and support for both regular SD and SDXL models, plus **long‑prompt support**.

Run:

    python main_v2.py [--xl] [--steps STEPS] [--cfg CFG] [--count NUM] 
                     [--width WIDTH] [--height HEIGHT]

The script loads a Stable Diffusion checkpoint, uses a fixed underground‑vault
prompt (well over the 77‑token limit), converts that prompt into *embeddings*
with **Compel** so nothing is truncated, and saves the resulting PNG under
*outputs/*.  Edit the CONSTANTS section to tweak anything.

Requirements (install once):

    pip install torch --extra-index-url https://download.pytorch.org/whl/cu121  # or "cpu" wheel
    pip install diffusers transformers accelerate safetensors compel

You can choose between regular Stable Diffusion and SDXL models using the --xl flag.
"""

from __future__ import annotations

import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Union, Tuple, Any

import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler
)
from compel import Compel
try:
    from compel.stable_diffusion_xl import CompelXL
except ImportError:
    # For older compel versions that don't support SDXL
    CompelXL = None

# -----------------------------------------------------------------------------
# Constants — tweak here
# -----------------------------------------------------------------------------
SD_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_STEPS = 35               # diffusion steps
DEFAULT_CFG_SCALE = 8.0          # guidance scale
DEFAULT_WIDTH_SD = 1344          # must be multiples of 8
DEFAULT_HEIGHT_SD = 768          # must be multiples of 8
DEFAULT_WIDTH_XL = 1024          # SDXL preferred resolution
DEFAULT_HEIGHT_XL = 1024         # SDXL preferred resolution
DEFAULT_NUM_IMAGES = 1           # number of images to generate
OUTPUT_DIR = Path("outputs")

DEFAULT_PROMPT = (
    "A vast, subterranean cathedral lost to myth — carved from obsidian and bone, "
    "drowned in perpetual night. No daylight intrudes; only the baleful crimson "
    "glow of a jagged Worldstone levitates over a spiked black dais, lava‑veined "
    "facets imprisoned inside a crackling sphere of eldritch energy. Four titanic "
    "iron chains, thick as trees, groan as they tether the sphere to the earth. "
    "Colossal statues flank the relic: on the left, Inarius the fallen angel, face "
    "shrouded by a martyr's hood, ghost‑fire wings spread behind baroque plate; on "
    "the right, Lilith, Queen of Hatred, crowned by sweeping ram‑horns, draped in "
    "shadow‑soaked silks, stone eyes smouldering ember‑red. Runes scar every column; "
    "braziers sputter sickly orange flames that paint restless shadows across ash‑thick "
    "air. Camera: ultra‑wide 32:9, low‑angle hero shot with dramatic perspective, "
    "anamorphic lens flares and subtle film grain. Lighting: cinematic chiaroscuro — "
    "deep blacks, pools of molten red, volumetric fog and drifting embers caught in "
    "shafts of infernal light. Style: grimdark high‑fantasy cinematic concept art, "
    "hyper‑detailed, 8k, Octane render, high dynamic range."
)

NEGATIVE_PROMPT = (
    "daylight, happy, bright, soft, cartoon, saturated colors, blurry, low‑res, extra "
    "characters, modern objects, text, watermark"
)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    parser.add_argument("--xl", action="store_true", help="Use SDXL model instead of regular SD")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of diffusion steps")
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG_SCALE, help="Guidance scale (how closely to follow the prompt)")
    parser.add_argument("--count", type=int, default=DEFAULT_NUM_IMAGES, help="Number of images to generate")
    parser.add_argument("--width", type=int, help="Image width (default depends on model)")
    parser.add_argument("--height", type=int, help="Image height (default depends on model)")
    
    return parser.parse_args()

def load_pipeline(use_xl: bool = False) -> Union[
    Tuple[StableDiffusionPipeline, Compel], 
    Tuple[StableDiffusionXLPipeline, Any]
]:
    """Load Stable Diffusion or SDXL pipeline with Compel helper for long prompts."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_xl:
        print(f"Loading SDXL model: {SDXL_MODEL_ID}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            pipe.enable_xformers_memory_efficient_attention()
        
        # No CompelXL, just return None as the compel processor
        return pipe, None
    else:
        print(f"Loading SD model: {SD_MODEL_ID}")
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()

        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    
    return pipe, compel_proc


def save_images(images: List["PIL.Image.Image"], seed: int, use_xl: bool) -> List[Path]:
    """Save generated images with appropriate naming"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "sdxl" if use_xl else "sd"
    paths = []
    for i, img in enumerate(images):
        path = OUTPUT_DIR / f"{prefix}_{timestamp}_{seed}_{i}.png"
        img.save(path)
        paths.append(path)
    return paths

# -----------------------------------------------------------------------------
# Main routine with command-line arguments
# -----------------------------------------------------------------------------

def main():
    args = parse_arguments()
    use_xl = args.xl
    
    # Set appropriate image dimensions based on model type and user input
    if args.width and args.height:
        width, height = args.width, args.height
    else:
        if use_xl:
            width, height = DEFAULT_WIDTH_XL, DEFAULT_HEIGHT_XL
        else:
            width, height = DEFAULT_WIDTH_SD, DEFAULT_HEIGHT_SD
    
    # Ensure dimensions are multiples of 8
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    seed = random.randrange(1 << 30)
    print(f"Using seed: {seed}")
    print(f"Generating {args.count} {'SDXL' if use_xl else 'SD'} image(s) at {width}x{height} resolution")
    
    pipe, compel_proc = load_pipeline(use_xl)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    print("Encoding long prompt with Compel (avoids token truncation)…")
    if use_xl:
        # For SDXL, we'll use the pipeline directly without Compel
        print("Generating image(s)… ☕")
        result = pipe(
            prompt=DEFAULT_PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_images_per_prompt=args.count,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=generator,
        )
    else:
        # Regular SD prompt handling
        prompt_embeds = compel_proc(DEFAULT_PROMPT)
        negative_embeds = compel_proc(NEGATIVE_PROMPT)
        
        print("Generating image(s)… ☕")
        result = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            width=width,
            height=height,
            num_images_per_prompt=args.count,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=generator,
        )

    paths = save_images(result.images, seed, use_xl)
    print("\n✅ Generation complete!")
    print(f"Model: {'SDXL' if use_xl else 'SD'}")
    print(f"Resolution: {width}x{height}")
    print(f"Steps: {args.steps}, CFG: {args.cfg}")
    print(f"Seed: {seed}")
    for p in paths:
        print(f"Saved: {p.resolve()}")


if __name__ == "__main__":
    main() 