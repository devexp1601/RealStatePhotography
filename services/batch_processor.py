
import os
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Any
from models.clip_classifier import classifier

 
class JobManager:
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    def create_job(self, file_count: int) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "file_count": file_count,
            "progress": {"completed": 0, "total": file_count, "percentage": 0}
        }
        return job_id
    
    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        return list(self.jobs.values())
    
    def get_active_count(self) -> int:
        return len([j for j in self.jobs.values() if j.get("status") == "processing"])
    
    def process_batch(self, job_id: str, file_paths: List[str]):
        job = self.jobs.get(job_id)
        if not job:
            return
        
        job["status"] = "processing"
        job["started_at"] = datetime.now().isoformat()
        
        results = []
        total = len(file_paths)
        
        for i, filepath in enumerate(file_paths):
            try:
                result = classifier.classify(filepath)
                result["filename"] = os.path.basename(filepath)
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": os.path.basename(filepath),
                    "status": "error",
                    "error": str(e)
                })
            
            job["progress"] = {
                "completed": i + 1,
                "total": total,
                "percentage": round((i + 1) / total * 100, 1)
            }
        
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        job["results"] = results
        
        success_count = sum(1 for r in results if r.get("status") == "success")
        interior_count = sum(1 for r in results if r.get("classification") == "interior")
        exterior_count = sum(1 for r in results if r.get("classification") == "exterior")
        
        job["summary"] = {
            "total": total,
            "success": success_count,
            "failed": total - success_count,
            "interior": interior_count,
            "exterior": exterior_count
        }
    
    def start_batch_processing(self, job_id: str, file_paths: List[str]):
        thread = threading.Thread(
            target=self.process_batch,
            args=(job_id, file_paths)
        )
        thread.start()
        return thread

job_manager = JobManager()
