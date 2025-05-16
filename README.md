# Stable Diffusion Image Generator

A Python script that generates fantasy-themed underground vault scenes using Stable Diffusion. This project demonstrates how to use the Diffusers library with Compel for long-prompt support, generating high-quality AI images beyond the standard token limit.

## Features

- Generates detailed fantasy underground vault scenes
- **Supports both regular Stable Diffusion and SDXL models**
- Supports prompts longer than the typical 77-token limit using Compel
- Configurable image dimensions, quality settings, and batch generation
- Command-line interface for easy customization
- Saves outputs with model type, timestamp and seed information for reproducibility

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
   git clone https://github.com/magyargergo/stable-diffusion-image-generator.git
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

### Basic Usage

Run the script with:

```
python main_v2.py
```

### Using SDXL

To use the SDXL model instead of regular Stable Diffusion:

```
python main_v2.py --xl
```

### Command-line Options

The script supports various command-line options:

```
python main_v2.py [--xl] [--steps STEPS] [--cfg CFG] [--count NUM] [--width WIDTH] [--height HEIGHT]
```

- `--xl`: Use SDXL model instead of regular SD
- `--steps`: Number of diffusion steps (default: 35)
- `--cfg`: Guidance scale (default: 8.0)
- `--count`: Number of images to generate (default: 1)
- `--width`: Custom image width
- `--height`: Custom image height

Generated images will be saved in the `outputs/` directory with filenames that include model type, timestamp and seed information.

## Configuration

You can customize the generation by editing the constants at the top of `main_v2.py`:

- `SD_MODEL_ID`: The regular Stable Diffusion model to use
- `SDXL_MODEL_ID`: The SDXL model to use
- `DEFAULT_STEPS`: Default number of diffusion steps
- `DEFAULT_CFG_SCALE`: Default guidance scale
- `DEFAULT_WIDTH_SD/XL`: Default image width for each model type
- `DEFAULT_HEIGHT_SD/XL`: Default image height for each model type
- `DEFAULT_PROMPT`: The prompt text describing the scene
- `NEGATIVE_PROMPT`: Elements to avoid in the generation

## License

MIT
