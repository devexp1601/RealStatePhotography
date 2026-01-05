import os
import uuid
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, RAW_EXTENSIONS
from services.pipeline import pipeline

pipeline_bp = Blueprint('pipeline', __name__)


def allowed_file(filename: str) -> bool:
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in (ALLOWED_EXTENSIONS | RAW_EXTENSIONS)


@pipeline_bp.route('/pipeline/process', methods=['POST'])
def process_pipeline():
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({"error": "No files selected"}), 400
    
    valid_counts = [1, 3, 5, 7]
    if len(files) not in valid_counts:
        return jsonify({
            "error": f"Expected 1, 3, 5, or 7 images, got {len(files)}. "
                     f"Use 1 for single image, 3/5/7 for HDR brackets."
        }), 400
    temp_paths = []
    try:
        for file in files:
            if file.filename and allowed_file(file.filename):
                filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                temp_paths.append(filepath)
        
        if len(temp_paths) == 0:
            return jsonify({"error": "No valid image files provided"}), 400
        
        if len(temp_paths) != len(files):
            return jsonify({
                "error": f"Some files were invalid. Got {len(temp_paths)} valid out of {len(files)}"
            }), 400
        
        temp_paths.sort()
        
        skip_hdr = request.form.get('skip_hdr', 'false').lower() == 'true'
        skip_enhance = request.form.get('skip_enhance', 'false').lower() == 'true'
        skip_sky = request.form.get('skip_sky', 'false').lower() == 'true'
        result = pipeline.process(temp_paths, skip_hdr=skip_hdr, skip_enhance=skip_enhance, skip_sky=skip_sky)
        response = result.to_dict()
        
        if result.output_path and os.path.exists(result.output_path):
            response["download_url"] = f"/pipeline/download/{os.path.basename(result.output_path)}"
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        for path in temp_paths:
            try:
                if path != result.output_path:
                    os.remove(path)
            except:
                pass


@pipeline_bp.route('/pipeline/download/<filename>', methods=['GET'])
def download_result(filename: str):
    from config import HDR_OUTPUT_FOLDER, OUTPUT_FOLDER, ENHANCED_OUTPUT_FOLDER
    
    for folder in [HDR_OUTPUT_FOLDER, ENHANCED_OUTPUT_FOLDER, OUTPUT_FOLDER, UPLOAD_FOLDER]:
        filepath = os.path.join(folder, secure_filename(filename))
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
    
    return jsonify({"error": "File not found"}), 404


@pipeline_bp.route('/pipeline/info', methods=['GET'])
def pipeline_info():
    return jsonify(pipeline.get_pipeline_info())
