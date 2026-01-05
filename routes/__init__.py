# Routes package
from routes.classify import classify_bp
from routes.jobs import jobs_bp
from routes.watcher import watcher_bp
from routes.hdr import hdr_bp
from routes.pipeline import pipeline_bp
from routes.enhance import enhance_bp
from routes.sky import sky_bp

__all__ = ['classify_bp', 'jobs_bp', 'watcher_bp', 'hdr_bp', 'pipeline_bp', 'enhance_bp', 'sky_bp']
