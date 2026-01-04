"""
Job management API routes
"""

from flask import Blueprint, jsonify

from services.batch_processor import job_manager

jobs_bp = Blueprint('jobs', __name__)


@jobs_bp.route('/job/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """Get the status of a batch processing job"""
    job = job_manager.get_job(job_id)
    
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify(job)


@jobs_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify({
        "jobs": job_manager.list_jobs(),
        "total": len(job_manager.list_jobs())
    })
