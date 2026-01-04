import os
import math
from typing import Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image
import torch

from config import HDR_OUTPUT_FOLDER
from utils.logger import setup_logger

log = setup_logger('enhancer')

DIFFUSERS_AVAILABLE = None


@dataclass
class TileConfig:
    tile_size: int = 1024
    overlap: int = 128
    min_tile_size: int = 512 


@dataclass 
class EnhanceConfig:
    strength: float = 0.3  
    controlnet_scale: float = 0.8  
    guidance_scale: float = 7.0
    num_inference_steps: int = 20  
    prompt: str = "professional real estate photography, well-lit, vibrant colors, high quality"
    negative_prompt: str = "blurry, dark, underexposed, overexposed, artificial, fake"


class TiledEnhancer:
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tile_config = TileConfig()
        self.enhance_config = EnhanceConfig()
        self._models_loaded = False
        self._diffusers_available = None
        
    @property
    def is_available(self) -> bool:
        if self._diffusers_available is None:
            try:
                import diffusers
                self._diffusers_available = True
            except ImportError:
                self._diffusers_available = False
        return self._diffusers_available
    
    @property
    def is_loaded(self) -> bool:
        return self._models_loaded
    
    def load_models(self):
        if not self.is_available:
            raise RuntimeError("diffusers not installed. Run: pip install diffusers accelerate safetensors")
        
        if self._models_loaded:
            log.info("Models already loaded")
            return
        
        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            ControlNetModel,
        )
        
        log.info("=" * 50)
        log.info("Loading SDXL + ControlNet models...")
        log.info("This may take a while on first run (~12GB download)")
        log.info("=" * 50)
        
        start_time = datetime.now()
        
        log.info("Loading ControlNet Tile...")
        controlnet_tile = ControlNetModel.from_pretrained(
            "xinsir/controlnet-tile-sdxl-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        log.info("Loading ControlNet Canny...")
        controlnet_canny = ControlNetModel.from_pretrained(
            "xinsir/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        log.info("Loading SDXL Base...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=[controlnet_tile, controlnet_canny],
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        self.pipe = self.pipe.to(self.device)
        
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                log.warning(f"xformers not available: {e}")
        
        self._models_loaded = True
        
        elapsed = (datetime.now() - start_time).total_seconds()
        log.info(f"✅ Models loaded in {elapsed:.1f}s")
    
    def _extract_canny(self, image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def _calculate_tiles(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        tile_size = self.tile_config.tile_size
        overlap = self.tile_config.overlap
        step = tile_size - overlap
        
        tiles = []
        
        y = 0
        while y < height:
            x = 0
            tile_h = min(tile_size, height - y)
            
            while x < width:
                tile_w = min(tile_size, width - x)
                tiles.append((x, y, tile_w, tile_h))
                
                if x + tile_size >= width:
                    break
                x += step
            
            if y + tile_size >= height:
                break
            y += step
        
        return tiles
    
    def _create_blend_mask(self, width: int, height: int, overlap: int) -> np.ndarray:
        mask = np.ones((height, width), dtype=np.float32)
        
        if overlap > 0:
            for i in range(overlap):
                alpha = float(i) / overlap
                # Left edge
                mask[:, i] = np.minimum(mask[:, i], alpha)
                # Right edge
                mask[:, -(i+1)] = np.minimum(mask[:, -(i+1)], alpha)
                # Top edge
                mask[i, :] = np.minimum(mask[i, :], alpha)
                # Bottom edge
                mask[-(i+1), :] = np.minimum(mask[-(i+1), :], alpha)
        
        return mask
    
    def enhance_tile(
        self, 
        tile: np.ndarray,
        config: Optional[EnhanceConfig] = None
    ) -> np.ndarray:
        
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        config = config or self.enhance_config
        
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile_pil = Image.fromarray(tile_rgb)
        
        original_size = tile_pil.size
        tile_pil = tile_pil.resize((1024, 1024), Image.LANCZOS)
        
        tile_resized = np.array(tile_pil)
        canny_image = self._extract_canny(cv2.cvtColor(tile_resized, cv2.COLOR_RGB2BGR))
        canny_pil = Image.fromarray(cv2.cvtColor(canny_image, cv2.COLOR_BGR2RGB))
        
        with torch.no_grad():
            result = self.pipe(
                prompt=config.prompt,
                negative_prompt=config.negative_prompt,
                image=[tile_pil, canny_pil],  # List of control images for each ControlNet
                controlnet_conditioning_scale=[config.controlnet_scale, config.controlnet_scale * 0.5],
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
            ).images[0]
        
        result = result.resize(original_size, Image.LANCZOS)
        
        result_np = np.array(result)
        return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    
    def enhance_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        config: Optional[EnhanceConfig] = None
    ) -> dict:

        
        if not self._models_loaded:
            self.load_models()
        
        config = config or self.enhance_config
        
        log.info("=" * 50)
        log.info(f"🎨 ENHANCING: {os.path.basename(image_path)}")
        log.info("=" * 50)
        
        start_time = datetime.now()
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        log.info(f"Image size: {width}x{height}")
        
        tiles = self._calculate_tiles(width, height)
        log.info(f"Processing {len(tiles)} tiles...")
        
        output = np.zeros_like(image, dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        for i, (x, y, w, h) in enumerate(tiles):
            log.info(f"  Tile {i+1}/{len(tiles)} at ({x},{y}) size {w}x{h}")
            
            tile = image[y:y+h, x:x+w].copy()
            
            if w < self.tile_config.tile_size or h < self.tile_config.tile_size:
                padded = np.zeros((self.tile_config.tile_size, self.tile_config.tile_size, 3), dtype=np.uint8)
                padded[:h, :w] = tile
                tile = padded
            
            enhanced_tile = self.enhance_tile(tile, config)
            
            enhanced_tile = enhanced_tile[:h, :w]
            
            blend_mask = self._create_blend_mask(w, h, self.tile_config.overlap)
            
            for c in range(3):
                output[y:y+h, x:x+w, c] += enhanced_tile[:, :, c].astype(np.float32) * blend_mask
            weights[y:y+h, x:x+w] += blend_mask
        
        weights = np.maximum(weights, 1e-6)
        for c in range(3):
            output[:, :, c] /= weights
        
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        if output_path is None:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(HDR_OUTPUT_FOLDER, f"{basename}_enhanced.jpg")
        
        cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        elapsed = (datetime.now() - start_time).total_seconds()
        log.info(f"✅ Enhanced in {elapsed:.1f}s → {output_path}")
        
        return {
            "status": "success",
            "input_path": image_path,
            "output_path": output_path,
            "dimensions": {"width": width, "height": height},
            "tiles_processed": len(tiles),
            "processing_time_s": round(elapsed, 1)
        }
    
    def get_info(self) -> dict:
        return {
            "available": self.is_available,
            "models_loaded": self._models_loaded,
            "device": self.device,
            "tile_config": {
                "tile_size": self.tile_config.tile_size,
                "overlap": self.tile_config.overlap
            },
            "enhance_config": {
                "strength": self.enhance_config.strength,
                "controlnet_scale": self.enhance_config.controlnet_scale,
                "steps": self.enhance_config.num_inference_steps
            }
        }


# Global singleton
enhancer = TiledEnhancer()
