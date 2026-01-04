"""
HDR Processing API Routes
"""

import os
import uuid
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, HDR_OUTPUT_FOLDER, ALLOWED_EXTENSIONS, RAW_EXTENSIONS
from services.hdr_processor import hdr_processor

hdr_bp = Blueprint('hdr', __name__)


def allowed_hdr_file(filename: str) -> bool:
    """Check if file extension is allowed for HDR processing"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in (ALLOWED_EXTENSIONS | RAW_EXTENSIONS)


@hdr_bp.route('/hdr/merge', methods=['POST'])
def merge_brackets():
    """
    Merge HDR brackets from uploaded images
    
    Accepts: 1, 3, 5, or 7 images as multipart form data
    Returns: Merged HDR image info with download URL
    """
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({"error": "No files selected"}), 400
    
    if len(files) not in [1, 3, 5, 7]:
        return jsonify({
            "error": f"Expected 1, 3, 5, or 7 bracket images, got {len(files)}"
        }), 400
    
    # Save uploaded files temporarily
    temp_paths = []
    try:
        for file in files:
            if file.filename and allowed_hdr_file(file.filename):
                filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                temp_paths.append(filepath)
        
        if len(temp_paths) == 0:
            return jsonify({"error": "No valid image files provided"}), 400
        
        if len(temp_paths) != len(files):
            return jsonify({
                "error": f"Some files were invalid. Expected {len(files)}, got {len(temp_paths)}"
            }), 400
        
        # Sort by filename to ensure correct bracket order
        temp_paths.sort()
        
        # Process HDR merge
        result = hdr_processor.merge_brackets(temp_paths)
        
        # Add download URL
        result["download_url"] = f"/hdr/download/{os.path.basename(result['output_path'])}"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Cleanup temp files
        for path in temp_paths:
            try:
                os.remove(path)
            except:
                pass


@hdr_bp.route('/hdr/merge-zip', methods=['POST'])
def merge_from_zip():
    """
    Merge HDR brackets from uploaded ZIP file
    
    Accepts: ZIP file containing 1, 3, 5, or 7 images
    Returns: Merged HDR image info with download URL
    """
    if 'zip' not in request.files:
        return jsonify({"error": "No ZIP file provided"}), 400
    
    zip_file = request.files['zip']
    
    if not zip_file.filename or not zip_file.filename.lower().endswith('.zip'):
        return jsonify({"error": "Invalid ZIP file"}), 400
    
    # Save ZIP temporarily
    zip_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex[:8]}.zip")
    
    try:
        zip_file.save(zip_path)
        
        # Process HDR merge from ZIP
        result = hdr_processor.process_zip(zip_path)
        
        # Add download URL
        result["download_url"] = f"/hdr/download/{os.path.basename(result['output_path'])}"
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Cleanup ZIP
        try:
            os.remove(zip_path)
        except:
            pass


@hdr_bp.route('/hdr/download/<filename>', methods=['GET'])
def download_hdr_result(filename: str):
    """Download a merged HDR image"""
    filepath = os.path.join(HDR_OUTPUT_FOLDER, secure_filename(filename))
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(filepath, as_attachment=True)


@hdr_bp.route('/hdr/info', methods=['GET'])
def hdr_info():
    """Get HDR processing information"""
    return jsonify({
        "supported_brackets": [1, 3, 5, 7],
        "supported_formats": list(ALLOWED_EXTENSIONS | RAW_EXTENSIONS),
        "alignment_enabled": True,
        "algorithm": "Mertens Exposure Fusion",
        "output_folder": HDR_OUTPUT_FOLDER
    })
