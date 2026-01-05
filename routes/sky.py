"""
Sky Replacement API Routes
Endpoints for detecting and replacing skies in exterior photos.
"""

import os
import uuid
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, ENHANCED_OUTPUT_FOLDER, SKY_LIBRARY_FOLDER
from services.sky_replacer import sky_replacer, SkyConfig

sky_bp = Blueprint('sky', __name__)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@sky_bp.route('/sky/info', methods=['GET'])
def sky_info():
    """Get sky replacement service info and status"""
    return jsonify(sky_replacer.get_info())


@sky_bp.route('/sky/load-models', methods=['POST'])
def load_models():
    """
    Load sky replacement models (segmentation + inpainting).
    This downloads models on first run.
    """
    try:
        sky_replacer.load_models()
        return jsonify({
            "status": "success",
            "message": "Sky replacement models loaded",
            "device": sky_replacer.device
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@sky_bp.route('/sky/detect', methods=['POST'])
def detect_sky():
    """
    Detect sky region in an image (without replacing).
    Returns the sky mask and ratio.
    
    Form data:
        - image: Image file
    """
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
        import cv2
        
        # Load and detect
        image = cv2.imread(input_path)
        if image is None:
            return jsonify({"error": "Could not load image"}), 400
        
        sky_replacer.segmenter.load_model()
        mask, ratio = sky_replacer.segmenter.detect_sky(image)
        
        # Save mask
        mask_filename = f"{os.path.splitext(filename)[0]}_sky_mask.png"
        mask_path = os.path.join(ENHANCED_OUTPUT_FOLDER, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        return jsonify({
            "status": "success",
            "sky_ratio": round(ratio, 3),
            "sky_percentage": f"{ratio * 100:.1f}%",
            "mask_path": mask_path,
            "download_url": f"/sky/download/{mask_filename}",
            "recommendation": "replace" if 0.05 <= ratio <= 0.7 else "skip"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Cleanup
        try:
            os.remove(input_path)
        except:
            pass


@sky_bp.route('/sky/replace', methods=['POST'])
def replace_sky():
    """
    Replace sky in an exterior image using sky library.
    
    Form data:
        - image: Image file
        - sky_index: Optional int (0-20) to select specific sky
        - use_library: Optional bool (default true) - use library vs SDXL
        - feather_amount: Optional int (default 30) - edge blending pixels
    """
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
        # Get optional config overrides
        config = SkyConfig()
        
        # Sky library options
        use_library = request.form.get('use_library', 'true').lower() == 'true'
        config.use_library = use_library
        
        feather_amount = request.form.get('feather_amount', type=int)
        if feather_amount:
            config.feather_amount = max(0, min(100, feather_amount))
        
        # Get sky index (for specific sky selection)
        sky_index = request.form.get('sky_index', type=int)
        
        # SDXL fallback options
        inpaint_strength = request.form.get('inpaint_strength', type=float)
        if inpaint_strength:
            config.inpaint_strength = max(0.5, min(1.0, inpaint_strength))
        
        sky_prompt = request.form.get('sky_prompt')
        if sky_prompt:
            config.sky_prompt = sky_prompt
        
        # Replace sky
        result = sky_replacer.replace_sky(input_path, sky_index=sky_index, config=config)
        
        # Add download URL
        if result.get("status") == "success":
            result["download_url"] = f"/sky/download/{os.path.basename(result['output_path'])}"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Cleanup input file
        try:
            os.remove(input_path)
        except:
            pass


@sky_bp.route('/sky/library', methods=['GET'])
def list_sky_library():
    """List all available sky images in the library"""
    sky_replacer.sky_library.load()
    return jsonify({
        "available": sky_replacer.sky_library.available,
        "count": sky_replacer.sky_library.count,
        "skies": sky_replacer.sky_library.list_skies()
    })


@sky_bp.route('/sky/library/<int:index>', methods=['GET'])
def preview_sky(index: int):
    """Preview a specific sky image from the library"""
    sky_replacer.sky_library.load()
    
    skies = sky_replacer.sky_library.list_skies()
    if not skies or index >= len(skies):
        return jsonify({"error": "Sky index out of range"}), 404
    
    sky_path = os.path.join(SKY_LIBRARY_FOLDER, skies[index])
    if os.path.exists(sky_path):
        return send_file(sky_path)
    
    return jsonify({"error": "Sky image not found"}), 404


@sky_bp.route('/sky/download/<filename>', methods=['GET'])
def download_sky_result(filename: str):
    """Download a sky replacement result"""
    filepath = os.path.join(ENHANCED_OUTPUT_FOLDER, secure_filename(filename))
    
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    
    return jsonify({"error": "File not found"}), 404
