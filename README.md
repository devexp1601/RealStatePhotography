# 🏠 Real Estate Photo Classifier

AI-powered image classification for real estate photography using OpenAI's CLIP model.

## Features

- **Interior vs Exterior Classification** - Automatically detect if a photo is inside or outside
- **Confidence Scores** - Get detailed probability scores for each category
- **GPU Acceleration** - Automatic CUDA detection for faster inference
- **Beautiful UI** - Modern dark theme with drag-and-drop support
- **REST API** - Easy integration with other services

## Quick Start

### 1. Create Virtual Environment

```bash
cd RealStatePhotography
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
python app.py
```

### 4. Open in Browser

Navigate to: **http://localhost:5000**

## API Endpoints

### Health Check
```bash
GET /health
```
Response:
```json
{
    "status": "healthy",
    "service": "Real Estate Photo Classifier",
    "gpu_available": true
}
```

### Classify Image (Upload)
```bash
POST /classify
Content-Type: multipart/form-data
Body: image=<file>
```
Response:
```json
{
    "classification": "interior",
    "confidence": 92.5,
    "scores": {
        "interior": 92.5,
        "exterior": 7.5
    },
    "filename": "living_room.jpg"
}
```

### Classify Image (URL)
```bash
POST /classify-url
Content-Type: application/json
Body: {"url": "https://example.com/house.jpg"}
```

## Tech Stack

- **Flask** - Web framework
- **CLIP (ViT-B/32)** - Image classification model
- **PyTorch** - Deep learning framework
- **Transformers** - Model loading (Hugging Face)

## Hardware Requirements

| Mode | Requirement |
|------|-------------|
| GPU | NVIDIA GPU with CUDA support |
| CPU | Works, but slower (~10x) |
| RAM | 4GB minimum |
| VRAM | 2GB minimum (for GPU mode) |

## Project Structure

```
RealStatePhotography/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web UI
├── uploads/            # Uploaded images (auto-created)
└── README.md           # This file
```

## Next Steps

This classifier is **Phase 1** of the Real Estate Photo Processing Pipeline. Upcoming:

- [ ] SDXL Lightning for image enhancement
- [ ] ControlNet for texture/geometry preservation
- [ ] Sky replacement for exterior photos
- [ ] Batch processing API
- [ ] AWS deployment

## License

MIT
