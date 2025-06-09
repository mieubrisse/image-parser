# Image Parser CLI

A privacy-focused CLI tool that analyzes images and provides detailed descriptions of their contents using local AI processing.

## Features

- Local image analysis using LLaVA (Large Language and Vision Assistant)
- Privacy-focused: all processing happens on your machine
- Fast processing with GPU acceleration (when available)
- Detailed descriptions of image contents
- Simple and intuitive CLI interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/image-parser.git
cd image-parser
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
# Analyze an image
python src/main.py analyze path/to/your/image.png

# Get help
python src/main.py --help
```

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (but not required)
- 8GB+ RAM recommended
- 10GB+ free disk space for model storage

## Performance Notes

- First run will download the LLaVA model (~7GB)
- Processing time depends on your hardware:
  - GPU: ~1-2 seconds per image
  - CPU: ~5-10 seconds per image
- Memory usage: ~4GB RAM

## License

MIT 