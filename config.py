
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
WATCH_FOLDER = os.path.join(BASE_DIR, 'watch_input')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'processed')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'tiff', 'tif'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

CLASSIFICATION_LABELS = [
    "an interior photo of a room inside a house",
    "an exterior photo of a house or building from outside"
]

WATCHER_SCAN_INTERVAL = 2

HDR_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'hdr_output')
HDR_ALIGNMENT_ENABLED = True
HDR_MIN_BRACKETS = 1
HDR_MAX_BRACKETS = 7

RAW_EXTENSIONS = {'cr2', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2'}

ENHANCED_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'enhanced_output')
TILE_SIZE = 1024
TILE_OVERLAP = 128
CONTROLNET_SCALE = 0.8

for folder in [UPLOAD_FOLDER, WATCH_FOLDER, OUTPUT_FOLDER, HDR_OUTPUT_FOLDER, ENHANCED_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)
