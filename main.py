# worldstone_vault_generator.py
"""Generate a subterranean Worldstone‑vault scene with Stable Diffusion – **no
command‑line arguments**, and now with **long‑prompt support**.

Run:

    python worldstone_vault_generator.py

The script loads a Stable Diffusion checkpoint, uses a fixed underground‑vault
prompt (well over the 77‑token limit), converts that prompt into *embeddings*
with **Compel** so nothing is truncated, and saves the resulting PNG under
*outputs/*.  Edit the CONSTANTS section to tweak anything.

Requirements (install once):

    pip install torch --extra-index-url https://download.pytorch.org/whl/cu118  # or "cpu" wheel
    pip install diffusers transformers accelerate safetensors compel

If you replace the model id with a more recent SDXL or SD‑2.1‑768 model, the
script still works – Compel simply builds longer embeddings.
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from compel import Compel

# -----------------------------------------------------------------------------
# Constants — tweak here
# -----------------------------------------------------------------------------
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
STEPS = 35               # diffusion steps
CFG_SCALE = 8.0          # guidance scale
WIDTH, HEIGHT = 1344, 768  # must be multiples of 8
NUM_IMAGES = 10      # number of images to generate
OUTPUT_DIR = Path("outputs")

DEFAULT_PROMPT = (
    "A vast, subterranean cathedral lost to myth — carved from obsidian and bone, "
    "drowned in perpetual night. No daylight intrudes; only the baleful crimson "
    "glow of a jagged Worldstone levitates over a spiked black dais, lava‑veined "
    "facets imprisoned inside a crackling sphere of eldritch energy. Four titanic "
    "iron chains, thick as trees, groan as they tether the sphere to the earth. "
    "Colossal statues flank the relic: on the left, Inarius the fallen angel, face "
    "shrouded by a martyr’s hood, ghost‑fire wings spread behind baroque plate; on "
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

def load_pipeline(model_id: str) -> tuple[StableDiffusionPipeline, Compel]:
    """Load Stable Diffusion and a Compel helper for long prompts."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()

    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    return pipe, compel_proc


def save_images(images: List["PIL.Image.Image"], seed: int) -> List[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for i, img in enumerate(images):
        path = OUTPUT_DIR / f"worldstone_{timestamp}_{seed}_{i}.png"
        img.save(path)
        paths.append(path)
    return paths

# -----------------------------------------------------------------------------
# Main routine (no arguments!)
# -----------------------------------------------------------------------------

def main():
    seed = random.randrange(1 << 30)
    print("Loading model (this can take a minute the first time)…")
    pipe, compel_proc = load_pipeline(MODEL_ID)

    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    print("Encoding long prompt with Compel (avoids 77‑token truncation)…")
    prompt_embeds = compel_proc(DEFAULT_PROMPT)
    negative_embeds = compel_proc(NEGATIVE_PROMPT)

    print("Generating image… ☕")
    result = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        width=WIDTH,
        height=HEIGHT,
        num_images_per_prompt=NUM_IMAGES,
        num_inference_steps=STEPS,
        guidance_scale=CFG_SCALE,
        generator=generator,
    )

    paths = save_images(result.images, seed)
    print("\n✅ Generation complete!")
    print("Seed:", seed)
    for p in paths:
        print("Saved:", p.resolve())


if __name__ == "__main__":
    main()
