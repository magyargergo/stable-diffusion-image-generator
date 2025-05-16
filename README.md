# Stable Diffusion Image Generator

A Python script that generates fantasy-themed underground vault scenes using Stable Diffusion. This project demonstrates how to use the Diffusers library with Compel for long-prompt support, generating high-quality AI images beyond the standard token limit.

## Features

- Generates detailed fantasy underground vault scenes
- Supports prompts longer than the typical 77-token limit using Compel
- Configurable image dimensions, quality settings, and batch generation
- Saves outputs with timestamp and seed information for reproducibility

## Requirements

- Python 3.8+
- PyTorch
- Diffusers
- Transformers
- Compel
- Other dependencies in requirements.txt

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stable-diffusion-image-generator.git
   cd stable-diffusion-image-generator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On Linux/macOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with:

```
python main.py
```

Generated images will be saved in the `outputs/` directory with filenames that include timestamp and seed information.

## Configuration

You can customize the generation by editing the constants at the top of `main.py`:

- `MODEL_ID`: The Stable Diffusion model to use
- `STEPS`: Number of diffusion steps (higher = more detailed but slower)
- `CFG_SCALE`: Guidance scale (how closely to follow the prompt)
- `WIDTH`, `HEIGHT`: Output image dimensions
- `NUM_IMAGES`: Number of images to generate per run
- `DEFAULT_PROMPT`: The prompt text describing the scene
- `NEGATIVE_PROMPT`: Elements to avoid in the generation

## License

MIT 