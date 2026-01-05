# 🏠 Real Estate Photo Processing Pipeline

AI-powered photo processing pipeline for real estate photography. Automatically classifies, enhances, and optimizes property photos.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **CLIP Classification** | Automatically detect interior vs exterior photos |
| **HDR Bracket Merging** | Merge 3/5/7 exposure brackets into single HDR image |
| **SDXL Enhancement** | AI-powered image enhancement with ControlNet |
| **Sky Replacement** | Replace overcast skies with beautiful blue skies |
| **Batch Processing** | Process multiple images with job queue |
| **Folder Watching** | Auto-process images dropped into watch folder |
| **Web UI** | Modern dark theme with drag-and-drop support |

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Open **http://localhost:5000** in your browser.

## 📡 API Endpoints

### Core Pipeline

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pipeline/process` | POST | Full pipeline: HDR → Classify → Enhance → Sky Replace |
| `/classify` | POST | Classify single image (interior/exterior) |
| `/classify-batch` | POST | Batch classify multiple images |

### HDR Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hdr/merge` | POST | Merge HDR brackets (3/5/7 images) |
| `/hdr/merge-zip` | POST | Merge HDR from ZIP file |

### Enhancement

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/enhance/process` | POST | SDXL + ControlNet enhancement |
| `/enhance/load-models` | POST | Pre-load enhancement models |

### Sky Replacement

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sky/replace` | POST | Replace sky in exterior image |
| `/sky/detect` | POST | Detect sky region only |
| `/sky/library` | GET | List available sky images |

### Utilities

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/job/<id>` | GET | Get batch job status |
| `/folder-watcher/start` | POST | Start folder watcher |

## 🔧 Pipeline Options

The web UI provides checkboxes to customize the pipeline:

- **HDR Merge** - Combine exposure brackets
- **SDXL Enhancement** - AI enhancement (requires GPU)
- **Sky Replacement** - Replace sky in exteriors

## 📁 Project Structure

```
RealStatePhotography/
├── app.py                 # Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── routes/                # API route handlers
│   ├── classify.py        # Classification endpoints
│   ├── hdr.py             # HDR merge endpoints
│   ├── enhance.py         # Enhancement endpoints
│   ├── sky.py             # Sky replacement endpoints
│   └── pipeline.py        # Full pipeline endpoint
├── services/              # Business logic
│   ├── classifier.py      # CLIP classification
│   ├── hdr_merger.py      # HDR bracket merging
│   ├── enhancer.py        # SDXL enhancement
│   ├── sky_replacer.py    # Sky detection & replacement
│   └── pipeline.py        # Pipeline orchestration
├── templates/
│   └── index.html         # Web UI
├── sky_library/           # Sky images for replacement
├── uploads/               # Uploaded images
├── hdr_output/            # HDR merge output
└── enhanced_output/       # Enhanced image output
```

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| GPU VRAM | 6GB | 8GB+ |
| Storage | 20GB | 50GB+ |

**Note:** SDXL enhancement requires NVIDIA GPU with CUDA. Other features work on CPU.

## 🖼️ Sky Library

Add sky images to `sky_library/` folder for sky replacement:
- Supports: JPG, PNG, WEBP
- Recommended: High-resolution blue sky images
- The system randomly selects from available skies

## 📦 Tech Stack

- **Flask** - Web framework
- **CLIP** - Image classification
- **Stable Diffusion XL** - Image enhancement
- **ControlNet** - Structure preservation
- **OpenCV** - Image processing
- **PyTorch** - Deep learning

## 📄 License

MIT
