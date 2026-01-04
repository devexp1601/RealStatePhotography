"""
Enhancement API Routes
Endpoints for SDXL + ControlNet image enhancement
"""

import os
import uuid
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, ENHANCED_OUTPUT_FOLDER
from services.enhancer import enhancer, EnhanceConfig

enhance_bp = Blueprint('enhance', __name__)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@enhance_bp.route('/enhance/info', methods=['GET'])
def enhance_info():
    """Get enhancement service info and status"""
    return jsonify(enhancer.get_info())


@enhance_bp.route('/enhance/load-models', methods=['POST'])
def load_models():
    """
    Load SDXL + ControlNet models
    This downloads ~12GB on first run
    """
    if not enhancer.is_available:
        return jsonify({
            "error": "diffusers not installed. Run: pip install diffusers accelerate safetensors"
        }), 400
    
    try:
        enhancer.load_models()
        return jsonify({
            "status": "success",
            "message": "Models loaded successfully",
            "device": enhancer.device
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@enhance_bp.route('/enhance/process', methods=['POST'])
def enhance_image():
    """
    Enhance a single image using SDXL + ControlNet
    
    Form data:
        - image: Image file
        - strength: Optional float 0.1-0.5 (default 0.3)
        - controlnet_scale: Optional float 0.5-1.0 (default 0.8)
    """
    if not enhancer.is_available:
        return jsonify({
            "error": "diffusers not installed. Run: pip install diffusers accelerate safetensors"
        }), 400
    
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    # Save uploaded file
    filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)
    
    try:
        # Get optional config
        strength = request.form.get('strength', type=float)
        controlnet_scale = request.form.get('controlnet_scale', type=float)
        
        config = None
        if strength or controlnet_scale:
            config = EnhanceConfig()
            if strength:
                config.strength = max(0.1, min(0.5, strength))
            if controlnet_scale:
                config.controlnet_scale = max(0.5, min(1.0, controlnet_scale))
        
        # Enhance image
        result = enhancer.enhance_image(input_path, config=config)
        
        # Add download URL
        result["download_url"] = f"/enhance/download/{os.path.basename(result['output_path'])}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Cleanup input file
        try:
            os.remove(input_path)
        except:
            pass


@enhance_bp.route('/enhance/download/<filename>', methods=['GET'])
def download_enhanced(filename: str):
    """Download an enhanced image"""
    filepath = os.path.join(ENHANCED_OUTPUT_FOLDER, secure_filename(filename))
    
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    
    return jsonify({"error": "File not found"}), 404
