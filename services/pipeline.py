"""
Pipeline Orchestrator
Chains all processing steps: HDR → Normalize → Classify → (future: SDXL, etc.)
"""

import os
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from config import HDR_OUTPUT_FOLDER, OUTPUT_FOLDER
from services.hdr_processor import hdr_processor
from services.enhancer import enhancer
from models.clip_classifier import classifier
from utils.logger import get_pipeline_logger

# Initialize logger
log = get_pipeline_logger()


class PipelineStage(Enum):
    """Pipeline processing stages"""
    PENDING = "pending"
    HDR_MERGE = "hdr_merge"
    NORMALIZE = "normalize"
    CLASSIFY = "classify"
    ENHANCE = "enhance"  # Future: SDXL
    SKY_REPLACE = "sky_replace"  # Future: Exterior
    WINDOW_PULL = "window_pull"  # Future: Interior
    QC_CHECK = "qc_check"  # Future
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class PipelineResult:
    """Result of pipeline processing"""
    status: str = "pending"
    stage: str = "pending"
    
    # Input info
    input_count: int = 0
    input_paths: List[str] = field(default_factory=list)
    
    # HDR result
    hdr_merged: bool = False
    hdr_output_path: Optional[str] = None
    
    # Classification result
    classification: Optional[str] = None  # "interior" or "exterior"
    confidence: Optional[float] = None
    scores: dict = field(default_factory=dict)
    
    # Final output
    output_path: Optional[str] = None
    dimensions: dict = field(default_factory=dict)
    
    # Processing info
    processing_time_ms: int = 0
    stages_completed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "stage": self.stage,
            "input_count": self.input_count,
            "hdr_merged": self.hdr_merged,
            "hdr_output_path": self.hdr_output_path,
            "classification": self.classification,
            "confidence": self.confidence,
            "scores": self.scores,
            "output_path": self.output_path,
            "dimensions": self.dimensions,
            "processing_time_ms": self.processing_time_ms,
            "stages_completed": self.stages_completed,
            "error": self.error
        }


class Pipeline:
    """
    Main pipeline orchestrator
    
    Chains: Input → HDR (optional) → Normalize → Classify → (future stages)
    """
    
    def process(
        self, 
        image_paths: List[str],
        skip_hdr: bool = False,
        skip_enhance: bool = False
    ) -> PipelineResult:
        """
        Process images through the full pipeline
        
        Args:
            image_paths: List of image paths (1 for single, 3/5/7 for brackets)
            skip_hdr: Skip HDR merging even if multiple images
            
        Returns:
            PipelineResult with all processing info
        """
        log.info("=" * 50)
        log.info(f"🚀 PIPELINE START | {len(image_paths)} image(s)")
        log.info("=" * 50)
        
        start_time = datetime.now()
        result = PipelineResult(
            input_count=len(image_paths),
            input_paths=image_paths
        )
        
        try:
            # ===== STAGE 1: HDR MERGE (if applicable) =====
            result.stage = PipelineStage.HDR_MERGE.value
            
            if len(image_paths) > 1 and not skip_hdr:
                log.info(f"📸 STAGE 1: HDR MERGE | {len(image_paths)} brackets")
                hdr_start = datetime.now()
                
                # Multiple images - do HDR merge
                hdr_result = hdr_processor.merge_brackets(image_paths)
                result.hdr_merged = True
                result.hdr_output_path = hdr_result["output_path"]
                result.dimensions = hdr_result.get("dimensions", {})
                working_image = hdr_result["output_path"]
                result.stages_completed.append("hdr_merge")
                
                hdr_time = (datetime.now() - hdr_start).total_seconds() * 1000
                log.info(f"   ✓ HDR merged in {hdr_time:.0f}ms → {os.path.basename(working_image)}")
            else:
                log.info("📸 STAGE 1: PASSTHROUGH | Single image, skipping HDR")
                # Single image - pass through
                working_image = image_paths[0]
                result.hdr_merged = False
                result.stages_completed.append("passthrough")
            
            # ===== STAGE 2: NORMALIZE =====
            result.stage = PipelineStage.NORMALIZE.value
            log.info("🔧 STAGE 2: NORMALIZE | Standardizing image")
            # For now, HDR processor already normalizes
            # Future: Add explicit normalization step if needed
            result.stages_completed.append("normalize")
            log.info("   ✓ Normalization complete")
            
            # ===== STAGE 3: CLASSIFY =====
            result.stage = PipelineStage.CLASSIFY.value
            log.info("🏷️ STAGE 3: CLASSIFY | Running CLIP classification")
            classify_start = datetime.now()
            
            classify_result = classifier.classify(working_image)
            result.classification = classify_result["classification"]
            result.confidence = classify_result["confidence"]
            result.scores = classify_result.get("scores", {})
            if not result.dimensions:
                result.dimensions = classify_result.get("dimensions", {})
            result.stages_completed.append("classify")
            
            classify_time = (datetime.now() - classify_start).total_seconds() * 1000
            log.info(f"   ✓ Classified as {result.classification.upper()} ({result.confidence}%) in {classify_time:.0f}ms")
            
            # ===== STAGE 4: ENHANCE (if enabled) =====
            result.stage = PipelineStage.ENHANCE.value
            
            if skip_enhance:
                log.info("🎨 STAGE 4: ENHANCE | Skipped by request")
                result.stages_completed.append("enhance_skipped")
            elif not enhancer.is_available:
                log.info("🎨 STAGE 4: ENHANCE | Skipped (diffusers not installed)")
                result.stages_completed.append("enhance_unavailable")
            else:
                log.info("🎨 STAGE 4: ENHANCE | SDXL + ControlNet processing")
                enhance_start = datetime.now()
                
                # Enhance the image
                enhance_result = enhancer.enhance_image(working_image)
                working_image = enhance_result["output_path"]
                result.stages_completed.append("enhance")
                
                enhance_time = (datetime.now() - enhance_start).total_seconds()
                log.info(f"   ✓ Enhanced in {enhance_time:.1f}s ({enhance_result.get('tiles_processed', 0)} tiles)")
            
            # Log classification branch (future stages)
            if result.classification == "interior":
                log.info("🏠 Branch: INTERIOR path (future: LoRA + Window Pull)")
            else:
                log.info("🌳 Branch: EXTERIOR path (future: Sky Replacement)")
            
            # ===== COMPLETE =====
            result.stage = PipelineStage.COMPLETE.value
            result.status = "success"
            result.output_path = working_image
            
        except Exception as e:
            log.error(f"Pipeline error: {str(e)}")
            result.stage = PipelineStage.ERROR.value
            result.status = "error"
            result.error = str(e)
        
        # Calculate processing time
        end_time = datetime.now()
        result.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        log.info("=" * 50)
        log.info(f"✨ PIPELINE COMPLETE | {result.processing_time_ms}ms | {result.classification} ({result.confidence}%)")
        log.info("=" * 50)
        
        return result
    
    def process_single(self, image_path: str) -> PipelineResult:
        """Convenience method for single image"""
        return self.process([image_path])
    
    def get_pipeline_info(self) -> dict:
        """Get information about the pipeline stages"""
        return {
            "name": "Real Estate Photo Pipeline",
            "version": "1.0",
            "stages": [
                {
                    "name": "HDR Merge",
                    "implemented": True,
                    "description": "Merges 3/5/7 exposure brackets into single HDR image"
                },
                {
                    "name": "Normalize",
                    "implemented": True,
                    "description": "Standardizes image format, color space, size"
                },
                {
                    "name": "Classify",
                    "implemented": True,
                    "description": "CLIP-based interior/exterior classification"
                },
                {
                    "name": "SDXL Enhancement",
                    "implemented": False,
                    "description": "AI-powered color correction and enhancement"
                },
                {
                    "name": "Sky Replacement",
                    "implemented": False,
                    "description": "Replaces overcast skies in exterior photos"
                },
                {
                    "name": "Window Pull",
                    "implemented": False,
                    "description": "Recovers blown-out window views in interior photos"
                },
                {
                    "name": "QC Check",
                    "implemented": False,
                    "description": "Automated quality validation"
                }
            ],
            "supported_brackets": [1, 3, 5, 7]
        }


# Global singleton
pipeline = Pipeline()
