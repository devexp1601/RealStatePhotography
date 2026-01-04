"""
Classification API routes
"""

import os
import uuid
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from models.clip_classifier import classifier
from services.batch_processor import job_manager

classify_bp = Blueprint('classify', __name__)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@classify_bp.route('/classify', methods=['POST'])
def classify_single():
    """Classify a single uploaded image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        result = classifier.classify(filepath)
        result["filename"] = filename
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@classify_bp.route('/classify-batch', methods=['POST'])
def classify_batch():
    """Classify multiple images at once"""
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({"error": "No files selected"}), 400
    
    # Save all files
    file_paths = []
    for file in files:
        if file.filename and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            file_paths.append(filepath)
    
    if len(file_paths) == 0:
        return jsonify({"error": "No valid image files provided"}), 400
    
    # Create job
    job_id = job_manager.create_job(len(file_paths))
    
    # Start processing in background
    job_manager.start_batch_processing(job_id, file_paths)
    
    return jsonify({
        "job_id": job_id,
        "message": f"Processing {len(file_paths)} images",
        "status_url": f"/job/{job_id}"
    })


@classify_bp.route('/classify-url', methods=['POST'])
def classify_from_url():
    """Classify an image from URL"""
    import requests
    
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    try:
        response = requests.get(data['url'], timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_url_image.jpg')
        image.save(temp_path)
        
        result = classifier.classify(temp_path)
        result["url"] = data['url']
        
        os.remove(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
