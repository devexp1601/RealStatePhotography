
import os
import json
import threading
import time
from datetime import datetime

from config import WATCH_FOLDER, OUTPUT_FOLDER, WATCHER_SCAN_INTERVAL, ALLOWED_EXTENSIONS
from models.clip_classifier import classifier


class FolderWatcher:
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.processed_files = set()
    
    @property
    def is_running(self) -> bool:
        return self.running
    
    @property
    def watch_folder(self) -> str:
        return os.path.abspath(WATCH_FOLDER)
    
    @property
    def output_folder(self) -> str:
        return os.path.abspath(OUTPUT_FOLDER)
    
    def _is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def _watch_loop(self):
        """Main watching loop"""
        while self.running:
            try:
                for filename in os.listdir(WATCH_FOLDER):
                    filepath = os.path.join(WATCH_FOLDER, filename)
                    
                    if filepath in self.processed_files or not os.path.isfile(filepath):
                        continue
                    
                    if not self._is_allowed_file(filename):
                        continue
                    
                    self.processed_files.add(filepath)
                    
                    try:
                        result = classifier.classify(filepath)
                        result["filename"] = filename
                        result["processed_at"] = datetime.now().isoformat()
                        result["source"] = "folder_watcher"
                        
                        result_filename = f"{os.path.splitext(filename)[0]}_result.json"
                        result_path = os.path.join(OUTPUT_FOLDER, result_filename)
                        
                        with open(result_path, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        print(f"✅ Processed: {filename} → {result['classification']} ({result['confidence']}%)")
                        
                    except Exception as e:
                        print(f"❌ Error processing {filename}: {e}")
                
                time.sleep(WATCHER_SCAN_INTERVAL)
                
            except Exception as e:
                print(f"Folder watcher error: {e}")
                time.sleep(5)
    
    def start(self) -> dict:
        if self.running:
            return {
                "message": "Folder watcher already running",
                "watch_folder": self.watch_folder
            }
        
        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()
        
        return {
            "message": "Folder watcher started",
            "watch_folder": self.watch_folder,
            "output_folder": self.output_folder
        }
    
    def stop(self) -> dict:
        if not self.running:
            return {"message": "Folder watcher not running"}
        
        self.running = False
        return {"message": "Folder watcher stopped"}
    
    def status(self) -> dict:
        return {
            "running": self.running,
            "watch_folder": self.watch_folder,
            "output_folder": self.output_folder,
            "processed_count": len(self.processed_files)
        }


folder_watcher = FolderWatcher()
