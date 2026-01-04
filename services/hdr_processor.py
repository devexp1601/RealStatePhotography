import os
import zipfile
import tempfile
import shutil
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import rawpy
    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False
    print("Warning: rawpy not installed. RAW file support disabled.")

from config import (
    UPLOAD_FOLDER, 
    ALLOWED_EXTENSIONS,
    HDR_OUTPUT_FOLDER,
    HDR_ALIGNMENT_ENABLED
)


class HDRProcessor:

    RAW_EXTENSIONS = {'cr2', 'nef', 'arw', 'dng', 'raf', 'orf', 'rw2'}
    
    def __init__(self):
        self.merge_mertens = cv2.createMergeMertens()
        self.align_mtb = cv2.createAlignMTB() if HDR_ALIGNMENT_ENABLED else None
    
    def _is_raw_file(self, filename: str) -> bool:
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        return ext in self.RAW_EXTENSIONS
    
    def _load_image(self, path: str) -> np.ndarray:
        if self._is_raw_file(path):
            if not RAW_SUPPORT:
                raise ValueError(f"RAW file support not available. Install rawpy.")
            
            with rawpy.imread(path) as raw:
                # Process RAW with default settings
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=8
                )
                # Convert RGB to BGR for OpenCV
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            # Standard image formats
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            return img
    
    def _normalize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        # Ensure 8-bit
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Resize if target size specified
        if target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def _align_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if not HDR_ALIGNMENT_ENABLED or self.align_mtb is None:
            return images
        
        # Convert to list if needed
        aligned = list(images)
        self.align_mtb.process(aligned, aligned)
        return aligned
    
    def merge_brackets(
        self, 
        image_paths: List[str],
        output_path: Optional[str] = None,
        align: bool = True
    ) -> dict:

        num_images = len(image_paths)
        
        if num_images == 0:
            raise ValueError("No images provided")
        
        if num_images == 1:
            img = self._load_image(image_paths[0])
            result = self._normalize_image(img)
            
            if output_path is None:
                basename = Path(image_paths[0]).stem
                output_path = os.path.join(HDR_OUTPUT_FOLDER, f"{basename}_normalized.jpg")
            
            cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return {
                "status": "success",
                "method": "passthrough",
                "input_count": 1,
                "output_path": output_path,
                "dimensions": {"width": result.shape[1], "height": result.shape[0]}
            }
        
        if num_images not in [3, 5, 7]:
            raise ValueError(f"Expected 1, 3, 5, or 7 brackets, got {num_images}")
        
        print(f"Loading {num_images} bracket images...")
        images = [self._load_image(p) for p in image_paths]
        
        base_shape = images[0].shape
        for i, img in enumerate(images[1:], 1):
            if img.shape != base_shape:
                images[i] = cv2.resize(img, (base_shape[1], base_shape[0]))
        
        if align and HDR_ALIGNMENT_ENABLED:
            print("Aligning images...")
            images = self._align_images(images)
        
        print("Merging brackets with Mertens Fusion...")
        
        images_float = [img.astype(np.float32) / 255.0 for img in images]
        
        fusion = self.merge_mertens.process(images_float)
        
        print(f"Fusion output range: min={fusion.min():.4f}, max={fusion.max():.4f}")
        
        fusion_min = fusion.min()
        fusion_max = fusion.max()
        
        if fusion_max > fusion_min:
            fusion_normalized = (fusion - fusion_min) / (fusion_max - fusion_min)
        else:
            fusion_normalized = fusion
        
        fusion_8bit = (fusion_normalized * 255).astype(np.uint8)
        
        print(f"Output range: min={fusion_8bit.min()}, max={fusion_8bit.max()}")
        
        if output_path is None:
            basename = Path(image_paths[0]).stem.rsplit('_', 1)[0]
            output_path = os.path.join(HDR_OUTPUT_FOLDER, f"{basename}_hdr_merged.jpg")
        
        cv2.imwrite(output_path, fusion_8bit, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved merged HDR: {output_path}")
        
        return {
            "status": "success",
            "method": "mertens_fusion",
            "input_count": num_images,
            "output_path": output_path,
            "dimensions": {"width": fusion_8bit.shape[1], "height": fusion_8bit.shape[0]},
            "aligned": align and HDR_ALIGNMENT_ENABLED
        }
    
    def process_folder(self, folder_path: str, output_path: Optional[str] = None) -> dict:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a valid folder: {folder_path}")
        
        all_extensions = ALLOWED_EXTENSIONS | self.RAW_EXTENSIONS
        image_files = []
        
        for f in sorted(os.listdir(folder_path)):
            ext = f.rsplit('.', 1)[-1].lower() if '.' in f else ''
            if ext in all_extensions:
                image_files.append(os.path.join(folder_path, f))
        
        if not image_files:
            raise ValueError(f"No valid images found in folder: {folder_path}")
        
        return self.merge_brackets(image_files, output_path)
    
    def process_zip(self, zip_path: str, output_path: Optional[str] = None) -> dict:
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"Not a valid ZIP file: {zip_path}")
        
        temp_dir = tempfile.mkdtemp(prefix="hdr_")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
            
            all_extensions = ALLOWED_EXTENSIONS | self.RAW_EXTENSIONS
            image_files = []
            
            for root, dirs, files in os.walk(temp_dir):
                for f in sorted(files):
                    ext = f.rsplit('.', 1)[-1].lower() if '.' in f else ''
                    if ext in all_extensions:
                        image_files.append(os.path.join(root, f))
            
            if not image_files:
                raise ValueError("No valid images found in ZIP file")
            
            return self.merge_brackets(image_files, output_path)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


hdr_processor = HDRProcessor()
