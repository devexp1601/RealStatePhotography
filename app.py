import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS

from config import UPLOAD_FOLDER, WATCH_FOLDER, OUTPUT_FOLDER, HDR_OUTPUT_FOLDER, MAX_CONTENT_LENGTH
from models.clip_classifier import classifier
from services.folder_watcher import folder_watcher
from services.batch_processor import job_manager
from routes import classify_bp, jobs_bp, watcher_bp, hdr_bp, pipeline_bp, enhance_bp


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
    
    app.register_blueprint(classify_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(watcher_bp)
    app.register_blueprint(hdr_bp)
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(enhance_bp)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "Real Estate Photo Classifier",
            "gpu_available": classifier.is_gpu_available,
            "folder_watcher": folder_watcher.is_running,
            "active_jobs": job_manager.get_active_count()
        })
    
    return app


app = create_app()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🏠 Real Estate Photo Classifier API")
    print("=" * 60)
    print(f"📍 Server: http://localhost:5000")
    print(f"🔧 GPU Available: {classifier.is_gpu_available}")
    print(f"📁 Upload Folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"👁️ Watch Folder: {os.path.abspath(WATCH_FOLDER)}")
    print(f"💾 Output Folder: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"🖼️ HDR Output: {os.path.abspath(HDR_OUTPUT_FOLDER)}")
    print("=" * 60)
    print("\n📌 API Endpoints:")
    print("  POST /pipeline/process  - Full pipeline (HDR→Classify)")
    print("  POST /enhance/process   - SDXL+ControlNet enhancement")
    print("  POST /enhance/load-models - Load AI models (~12GB)")
    print("  POST /classify          - Single image classify")
    print("  POST /classify-batch    - Multiple images classify")
    print("  GET  /job/<id>          - Check batch status")
    print("  POST /hdr/merge         - Merge HDR brackets")
    print("  POST /folder-watcher/start - Start watching folder")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
