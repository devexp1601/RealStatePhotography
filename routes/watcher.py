"""
Folder watcher API routes
"""

from flask import Blueprint, jsonify

from services.folder_watcher import folder_watcher

watcher_bp = Blueprint('watcher', __name__)


@watcher_bp.route('/folder-watcher/start', methods=['POST'])
def start_watcher():
    """Start the folder watcher"""
    result = folder_watcher.start()
    return jsonify(result)


@watcher_bp.route('/folder-watcher/stop', methods=['POST'])
def stop_watcher():
    """Stop the folder watcher"""
    result = folder_watcher.stop()
    return jsonify(result)


@watcher_bp.route('/folder-watcher/status', methods=['GET'])
def watcher_status():
    """Get folder watcher status"""
    return jsonify(folder_watcher.status())
