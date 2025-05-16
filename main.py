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
    "Ultra-detailed dark-fantasy concept art, Diablo II Act V aesthetic, photorealistic 4K resolution, 3:2 —  "
    "vast subterranean cathedral carved from black basalt, Gothic arches vanishing into darkness;  "
    "colossal BLOOD-RED WORLDSTONE shard dominates the foreground, jagged surface ablaze with magma-like fissures, "
    "suspended above a tiered, rune-etched stone dais;  "
    "demon-forged iron cradle encircling mid-section, massive CHAINS anchoring to platform;  "
    "transparent spherical ENERGY BARRIER surrounds the shard, crimson lightning veining across its surface;  "
    "half-shrouded in gloom, two monumental weather-worn granite statues guard the relic:  "
    "INARIUS on the left — hooded, faceless angel in flared plate armor with staff, broad feathered wings spread behind him;  "
    "LILITH on the right — elegant demoness with twin crown-horns, serene yet sinister visage, fitted cuirass, folded bat-wings;  "
    "rune-adorned bridges and stairs, ember-littered floor, dust motes swirling in angled shafts of light;  "
    "foreboding cinematic lighting, rich shadow interplay, volumetric haze, intricate textures on stone, metal, crystal;  "
    "epic, grim, high contrast, Octane render"
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

def chunk_and_process_prompt(prompt, max_length=75):
    """Break a long prompt into chunks that fit within token limits and process sequentially."""
    words = prompt.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        # Roughly estimate token count (this is approximate)
        word_length = len(word.split()) 
        
        if current_length + word_length > max_length:
            # This chunk is full, save it and start a new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            # Add word to current chunk
            current_chunk.append(word)
            current_length += word_length
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_chunked_prompt(pipe, prompt, max_length=75):
    """Process a long prompt by breaking it into chunks and keeping the most important parts."""
    # If the prompt is short enough, just return it as is
    if len(prompt.split()) <= max_length:
        return prompt
    
    # Split into chunks
    chunks = chunk_and_process_prompt(prompt, max_length)
    print(f"Prompt was split into {len(chunks)} chunks")
    
    # For simplicity, if we have multiple chunks, keep:
    # 1. The first chunk (usually contains the main subject)
    # 2. Key style words from the last chunk (usually contains style directions)
    
    # Extract style keywords from the last chunk
    style_chunk = chunks[-1]
    style_keywords = []
    
    # Common style-related keywords
    style_markers = ["style", "render", "detailed", "art", "quality", "lighting", "resolution", 
                    "cinematic", "dynamic", "range", "aesthetic", "perspective", "mood", "tone"]
    
    # Extract style-related phrases
    for marker in style_markers:
        if marker in style_chunk.lower():
            # Find the phrase containing this marker
            words = style_chunk.split()
            for i, word in enumerate(words):
                if marker.lower() in word.lower():
                    # Take a few words before and after as context
                    start = max(0, i-3)
                    end = min(len(words), i+4)
                    style_keywords.append(" ".join(words[start:end]))
    
    # Combine first chunk with style keywords
    processed_prompt = chunks[0]
    if style_keywords:
        processed_prompt += ", " + ", ".join(style_keywords)
    
    print(f"Original prompt length: {len(prompt.split())} words")
    print(f"Processed prompt length: {len(processed_prompt.split())} words")
    return processed_prompt

def encode_prompt_properly(pipe, prompt, negative_prompt=None):
    """Create proper embeddings for SDXL without truncating the prompt"""
    # Get tokenizers and text encoders
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder_1 = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    
    # Process prompt in segments to handle very long prompts
    max_length = tokenizer_1.model_max_length
    words = prompt.split()
    segments = []
    current_segment = []
    current_length = 0
    
    # Roughly estimate token counts and segment the prompt
    for word in words:
        # Crude token estimation
        word_tokens = len(tokenizer_1.tokenize(word))
        if current_length + word_tokens > max_length - 2:  # -2 for special tokens
            segments.append(" ".join(current_segment))
            current_segment = [word]
            current_length = word_tokens
        else:
            current_segment.append(word)
            current_length += word_tokens
    
    if current_segment:
        segments.append(" ".join(current_segment))
    
    print(f"Split prompt into {len(segments)} segments")
    
    # Process each segment
    device = text_encoder_1.device
    prompt_embeds_list_1 = []
    prompt_embeds_list_2 = []
    pooled_embeds_list = []
    
    for segment in segments:
        # Process through first text encoder
        text_inputs_1 = tokenizer_1(
            segment,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        # Process through second text encoder
        text_inputs_2 = tokenizer_2(
            segment,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            # Get embeddings from first text encoder (hidden_states)
            prompt_embeds_1 = text_encoder_1(
                text_inputs_1.input_ids,
                attention_mask=text_inputs_1.attention_mask,
            )[0]
            
            # Get embeddings from second text encoder (hidden_states and pooled output)
            text_encoder_2_output = text_encoder_2(
                text_inputs_2.input_ids,
                attention_mask=text_inputs_2.attention_mask,
            )
            prompt_embeds_2 = text_encoder_2_output[0]
            pooled_embeds = text_encoder_2_output[1]
            
            # Collect embeddings
            prompt_embeds_list_1.append(prompt_embeds_1)
            prompt_embeds_list_2.append(prompt_embeds_2)
            pooled_embeds_list.append(pooled_embeds)
    
    # Combine segments (average the embeddings)
    if len(segments) > 1:
        prompt_embeds_1 = torch.stack(prompt_embeds_list_1).mean(dim=0)
        prompt_embeds_2 = torch.stack(prompt_embeds_list_2).mean(dim=0)
        pooled_prompt_embeds = torch.stack(pooled_embeds_list).mean(dim=0)
    else:
        prompt_embeds_1 = prompt_embeds_list_1[0]
        prompt_embeds_2 = prompt_embeds_list_2[0]
        pooled_prompt_embeds = pooled_embeds_list[0]
    
    # Process negative prompt
    if negative_prompt:
        neg_text_inputs_1 = tokenizer_1(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        neg_text_inputs_2 = tokenizer_2(
            negative_prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            # Get hidden states
            neg_prompt_embeds_1 = text_encoder_1(
                neg_text_inputs_1.input_ids,
                attention_mask=neg_text_inputs_1.attention_mask,
            )[0]
            
            # Get hidden states and pooled output
            neg_text_encoder_2_output = text_encoder_2(
                neg_text_inputs_2.input_ids,
                attention_mask=neg_text_inputs_2.attention_mask,
            )
            neg_prompt_embeds_2 = neg_text_encoder_2_output[0]
            neg_pooled_prompt_embeds = neg_text_encoder_2_output[1]
    else:
        # Create empty tensors for negative prompt
        neg_prompt_embeds_1 = torch.zeros_like(prompt_embeds_1)
        neg_prompt_embeds_2 = torch.zeros_like(prompt_embeds_2)
        neg_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    
    # Concatenate embeddings from both text encoders
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    negative_prompt_embeds = torch.cat([neg_prompt_embeds_1, neg_prompt_embeds_2], dim=-1)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds

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
        # For SDXL, use our custom embedding approach
        print("Using custom embedding for SDXL to handle long prompts...")
        try:
            # Use proper embedding with segmentation
            prompt_embeds, negative_embeds, pooled_prompt_embeds, pooled_negative_embeds = encode_prompt_properly(
                pipe, DEFAULT_PROMPT, NEGATIVE_PROMPT
            )
            
            print("Generating image(s) with full embedded prompt... ☕")
            result = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                pooled_negative_prompt_embeds=pooled_negative_embeds,
                width=width,
                height=height,
                num_images_per_prompt=args.count,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=generator,
            )
        except Exception as e:
            print(f"Error with custom embedding: {e}")
            print("Falling back to chunked prompt processing")
            try:
                processed_prompt = process_chunked_prompt(pipe, DEFAULT_PROMPT)
                print("Generating image(s) with processed prompt... ☕")
                result = pipe(
                    prompt=processed_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=width,
                    height=height,
                    num_images_per_prompt=args.count,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    generator=generator,
                )
            except Exception as e2:
                print(f"Error with chunked prompt: {e2}")
                print("Falling back to direct prompt input (will truncate)")
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