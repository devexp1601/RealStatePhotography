"""
Sky Replacement Service
Detects and replaces overcast/dull skies in exterior photos with clear blue skies.
Uses sky library images for fast, consistent results.
Falls back to SDXL Inpainting if no library available.
"""

import os
import random
from typing import Optional, Tuple, List
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image
import torch

from config import HDR_OUTPUT_FOLDER, ENHANCED_OUTPUT_FOLDER, SKY_LIBRARY_FOLDER
from utils.logger import setup_logger

log = setup_logger('sky_replacer')


@dataclass
class SkyConfig:
    """Configuration for sky replacement"""
    # Sky detection
    sky_threshold: float = 0.5
    min_sky_ratio: float = 0.005  # Minimum sky area (0.5%) - lowered for partial sky through trees
    max_sky_ratio: float = 0.6   # Maximum sky area (60%)
    
    # Blending
    feather_amount: int = 30     # Pixels to feather mask edges
    color_match_strength: float = 0.4  # How much to match sky colors
    
    # Sky library
    use_library: bool = True     # Use sky library instead of SDXL
    random_sky: bool = True      # Randomly select sky (False = use first)
    
    # SDXL fallback settings
    inpaint_strength: float = 0.85
    guidance_scale: float = 7.5
    num_inference_steps: int = 25
    sky_prompt: str = "beautiful clear blue sky with soft white clouds, sunny day, photorealistic"
    negative_prompt: str = "dark, stormy, overcast, grey, purple, pink, artificial"


class SkyLibrary:
    """Manages pre-made sky images for replacement"""
    
    def __init__(self, library_path: str = SKY_LIBRARY_FOLDER):
        self.library_path = library_path
        self._sky_images: List[str] = []
        self._loaded = False
    
    def load(self):
        """Load list of available sky images"""
        if self._loaded:
            return
        
        if not os.path.exists(self.library_path):
            log.warning(f"Sky library folder not found: {self.library_path}")
            self._loaded = True
            return
        
        # Find all image files
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        for filename in os.listdir(self.library_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                self._sky_images.append(os.path.join(self.library_path, filename))
        
        self._sky_images.sort()
        self._loaded = True
        log.info(f"✅ Sky library loaded: {len(self._sky_images)} sky images")
    
    @property
    def available(self) -> bool:
        self.load()
        return len(self._sky_images) > 0
    
    @property
    def count(self) -> int:
        self.load()
        return len(self._sky_images)
    
    def get_sky(self, index: Optional[int] = None, random_select: bool = True) -> Optional[np.ndarray]:
        """
        Get a sky image from the library.
        
        Args:
            index: Specific sky index (0-based). If None, uses random or first.
            random_select: If True and index is None, pick random sky.
        
        Returns:
            Sky image as numpy array (BGR) or None if no skies available.
        """
        self.load()
        
        if not self._sky_images:
            return None
        
        if index is not None:
            idx = index % len(self._sky_images)
        elif random_select:
            idx = random.randint(0, len(self._sky_images) - 1)
        else:
            idx = 0
        
        sky_path = self._sky_images[idx]
        sky_img = cv2.imread(sky_path)
        
        if sky_img is None:
            log.warning(f"Could not load sky image: {sky_path}")
            return None
        
        log.info(f"Using sky: {os.path.basename(sky_path)}")
        return sky_img
    
    def list_skies(self) -> List[str]:
        """Get list of available sky image names"""
        self.load()
        return [os.path.basename(p) for p in self._sky_images]


class SkySegmenter:
    """Sky segmentation using color-based detection or AI model"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_loaded = False
        self._use_simple = True  # Start with simple, upgrade if model loads
    
    def load_model(self):
        """Try to load AI segmentation model"""
        if self._model_loaded:
            return
        
        try:
            from transformers import AutoModelForImageSegmentation, AutoProcessor
            
            log.info("Loading sky segmentation model (RMBG-1.4)...")
            self.processor = AutoProcessor.from_pretrained(
                "briaai/RMBG-1.4",
                trust_remote_code=True
            )
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-1.4",
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            self._use_simple = False
            log.info("✅ AI segmentation model loaded")
            
        except Exception as e:
            log.info(f"Using color-based sky detection (AI model unavailable: {e})")
            self._use_simple = True
        
        self._model_loaded = True
    
    def _detect_sky_simple(self, image: np.ndarray) -> np.ndarray:
        """Color-based sky detection with building exclusion using edge detection."""
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sky color detection
        blue_sky = cv2.inRange(hsv, (90, 20, 130), (130, 255, 255))
        grey_sky = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
        light_blue = cv2.inRange(hsv, (85, 15, 160), (135, 120, 255))
        sky_color_mask = cv2.bitwise_or(blue_sky, grey_sky)
        sky_color_mask = cv2.bitwise_or(sky_color_mask, light_blue)
        
        # Edge detection for building exclusion
        edges = cv2.Canny(gray, 60, 150)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        edge_zones = cv2.dilate(edges, edge_kernel)
        
        # Top region (0-25%): full trust
        top_mask = np.zeros((height, width), dtype=np.uint8)
        top_mask[0:int(height * 0.25), :] = 255
        top_sky = cv2.bitwise_and(sky_color_mask, top_mask)
        
        # Middle region (25-50%): with edge exclusion
        middle_mask = np.zeros((height, width), dtype=np.uint8)
        middle_mask[int(height * 0.25):int(height * 0.50), :] = 255
        middle_sky = cv2.bitwise_and(sky_color_mask, middle_mask)
        middle_sky = cv2.bitwise_and(middle_sky, cv2.bitwise_not(edge_zones))
        
        # Flood fill from top edge
        sky_from_top = np.zeros_like(sky_color_mask)
        for y in range(min(40, height // 12)):
            for x in range(0, width, 4):
                if sky_color_mask[y, x] > 0 and sky_from_top[y, x] == 0:
                    temp_mask = sky_color_mask.copy()
                    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
                    cv2.floodFill(temp_mask, flood_mask, (x, y), 128)
                    connected = (temp_mask == 128).astype(np.uint8) * 255
                    connected[int(height * 0.50):, :] = 0
                    sky_from_top = cv2.bitwise_or(sky_from_top, connected)
        
        # Combine all regions
        combined_mask = cv2.bitwise_or(top_sky, middle_sky)
        combined_mask = cv2.bitwise_or(combined_mask, sky_from_top)
        
        # Cleanup
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            final_mask = np.zeros_like(combined_mask)
            min_area = (height * width) * 0.00008
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y_pos, w, h = cv2.boundingRect(contour)
                if area > min_area and y_pos < height * 0.55:
                    cv2.drawContours(final_mask, [contour], -1, 255, -1)
            combined_mask = final_mask
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)
        return combined_mask
    
    def _detect_sky_model(self, image: np.ndarray) -> np.ndarray:
        """Use AI model to detect foreground, invert for sky"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        mask = outputs[0].squeeze().cpu().numpy()
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Invert to get sky (background)
        sky_mask = 1.0 - mask
        
        # Position weighting
        height, width = image.shape[:2]
        position_weight = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            weight = 1.0 - (y / height)
            position_weight[y, :] = weight ** 0.3
        
        sky_mask = sky_mask * position_weight
        sky_mask = (sky_mask > 0.4).astype(np.uint8) * 255
        
        return sky_mask
    
    def detect_sky(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect sky region in image.
        
        Returns:
            (sky_mask, sky_ratio) - Binary mask and fraction of image that is sky
        """
        self.load_model()
        
        if self._use_simple:
            mask = self._detect_sky_simple(image)
        else:
            mask = self._detect_sky_model(image)
        
        sky_ratio = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1])
        
        return mask, sky_ratio


class SkyReplacer:
    """
    Main sky replacement service.
    Uses sky library for fast replacement, SDXL inpainting as fallback.
    """
    
    def __init__(self):
        self.segmenter = SkySegmenter()
        self.sky_library = SkyLibrary()
        self.inpaint_pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = SkyConfig()
        self._models_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._models_loaded
    
    def load_models(self):
        """Load segmentation model and sky library"""
        if self._models_loaded:
            return
        
        log.info("=" * 50)
        log.info("Loading Sky Replacement service...")
        log.info("=" * 50)
        
        start_time = datetime.now()
        
        # Load segmenter
        self.segmenter.load_model()
        
        # Load sky library
        self.sky_library.load()
        
        if self.sky_library.available:
            log.info(f"✅ Sky library ready ({self.sky_library.count} skies)")
        else:
            log.warning("⚠️ No sky library - will use SDXL inpainting (slower)")
        
        self._models_loaded = True
        elapsed = (datetime.now() - start_time).total_seconds()
        log.info(f"✅ Sky replacement ready in {elapsed:.1f}s")
    
    def _load_inpainting_model(self):
        """Lazy load SDXL inpainting (only if needed)"""
        if self.inpaint_pipe is not None:
            return
        
        try:
            from diffusers import StableDiffusionXLInpaintPipeline
            
            log.info("Loading SDXL Inpainting (fallback)...")
            self.inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            self.inpaint_pipe.to(self.device)
            
            if self.device == "cuda":
                try:
                    self.inpaint_pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass
                    
        except Exception as e:
            log.error(f"Failed to load SDXL inpainting: {e}")
            raise
    
    def _feather_mask(self, mask: np.ndarray, amount: int) -> np.ndarray:
        """Apply feathering to mask edges for smooth blending"""
        if amount <= 0:
            return mask.astype(np.float32) / 255.0
        
        feathered = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), amount)
        return feathered / 255.0
    
    def _match_colors(self, sky: np.ndarray, original: np.ndarray, 
                      mask: np.ndarray, strength: float = 0.4) -> np.ndarray:
        """
        Match sky colors to the scene for natural blending.
        Adjusts brightness and color temperature.
        """
        # Get color stats from edge of non-sky region (horizon area)
        mask_binary = mask > 127
        
        # Dilate mask to get horizon region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        horizon_region = cv2.dilate(mask.astype(np.uint8), kernel) - mask
        horizon_mask = horizon_region > 127
        
        if np.sum(horizon_mask) < 100:
            # Not enough horizon pixels, use non-sky region
            horizon_mask = ~mask_binary
        
        if np.sum(horizon_mask) < 100:
            return sky
        
        # Convert to LAB
        sky_lab = cv2.cvtColor(sky, cv2.COLOR_BGR2LAB).astype(np.float32)
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Get statistics from horizon region of original
        orig_horizon_mean = np.mean(orig_lab[horizon_mask], axis=0)
        
        # Get sky stats
        sky_mask_bool = mask_binary
        if np.sum(sky_mask_bool) > 100:
            sky_mean = np.mean(sky_lab[sky_mask_bool], axis=0)
        else:
            sky_mean = np.mean(sky_lab, axis=(0, 1))
        
        # Adjust sky colors toward horizon colors (for natural blend)
        # Only adjust A and B channels (color), keep L (luminance) mostly intact
        for i in [1, 2]:  # A and B channels
            adjustment = (orig_horizon_mean[i] - sky_mean[i]) * strength
            sky_lab[:, :, i] = sky_lab[:, :, i] + adjustment
        
        # Slight luminance adjustment
        l_adjustment = (orig_horizon_mean[0] - sky_mean[0]) * (strength * 0.3)
        sky_lab[:, :, 0] = sky_lab[:, :, 0] + l_adjustment
        
        sky_lab = np.clip(sky_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sky_lab, cv2.COLOR_LAB2BGR)
    
    def _blend_sky_library(self, image: np.ndarray, mask: np.ndarray, 
                           sky_index: Optional[int] = None) -> np.ndarray:
        """Blend a sky from library into the image"""
        height, width = image.shape[:2]
        
        # Get sky image
        sky = self.sky_library.get_sky(
            index=sky_index, 
            random_select=self.config.random_sky
        )
        
        if sky is None:
            raise RuntimeError("No sky images available in library")
        
        # Resize sky to match image
        sky_h, sky_w = sky.shape[:2]
        
        # Calculate how to fit sky (may need to crop/scale)
        # Try to maintain aspect ratio and cover sky region
        scale = max(width / sky_w, height / sky_h)
        new_w = int(sky_w * scale)
        new_h = int(sky_h * scale)
        
        sky_resized = cv2.resize(sky, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center crop to match image size
        start_x = (new_w - width) // 2
        start_y = (new_h - height) // 2
        sky_cropped = sky_resized[start_y:start_y+height, start_x:start_x+width]
        
        # Ensure exact size match
        if sky_cropped.shape[:2] != (height, width):
            sky_cropped = cv2.resize(sky_cropped, (width, height))
        
        # Match colors
        sky_matched = self._match_colors(
            sky_cropped, image, mask, self.config.color_match_strength
        )
        
        # Feather mask
        mask_feathered = self._feather_mask(mask, self.config.feather_amount)
        mask_3ch = np.stack([mask_feathered] * 3, axis=-1)
        
        # Blend
        result = (sky_matched * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def _inpaint_sky_sdxl(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate new sky using SDXL inpainting (fallback)"""
        self._load_inpainting_model()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray(mask)
        
        orig_size = image_pil.size
        target_size = (1024, 1024)
        
        image_resized = image_pil.resize(target_size, Image.LANCZOS)
        mask_resized = mask_pil.resize(target_size, Image.NEAREST)
        
        with torch.no_grad():
            result = self.inpaint_pipe(
                prompt=self.config.sky_prompt,
                negative_prompt=self.config.negative_prompt,
                image=image_resized,
                mask_image=mask_resized,
                strength=self.config.inpaint_strength,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
            ).images[0]
        
        result = result.resize(orig_size, Image.LANCZOS)
        result_np = np.array(result)
        return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    
    def replace_sky(
        self, 
        image_path: str, 
        output_path: Optional[str] = None,
        sky_index: Optional[int] = None,
        config: Optional[SkyConfig] = None
    ) -> dict:
        """
        Replace sky in an exterior image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save result (auto-generated if None)
            sky_index: Specific sky image index (None = random)
            config: Sky replacement configuration
            
        Returns:
            dict with status, paths, and processing info
        """
        if not self._models_loaded:
            self.load_models()
        
        config = config or self.config
        
        log.info("=" * 50)
        log.info(f"☁️ SKY REPLACEMENT: {os.path.basename(image_path)}")
        log.info("=" * 50)
        
        start_time = datetime.now()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        log.info(f"Image size: {width}x{height}")
        
        # Detect sky
        log.info("Detecting sky region...")
        sky_mask, sky_ratio = self.segmenter.detect_sky(image)
        log.info(f"Sky ratio: {sky_ratio:.1%}")
        
        # Check if sky replacement is needed
        if sky_ratio < config.min_sky_ratio:
            log.info(f"Sky ratio ({sky_ratio:.1%}) below threshold. Skipping.")
            return {
                "status": "skipped",
                "reason": "insufficient_sky",
                "sky_ratio": round(sky_ratio, 3),
                "input_path": image_path,
                "output_path": image_path
            }
        
        if sky_ratio > config.max_sky_ratio:
            log.info(f"Sky ratio ({sky_ratio:.1%}) above maximum. Skipping.")
            return {
                "status": "skipped", 
                "reason": "too_much_sky",
                "sky_ratio": round(sky_ratio, 3),
                "input_path": image_path,
                "output_path": image_path
            }
        
        # Replace sky
        method = "library"
        if config.use_library and self.sky_library.available:
            log.info(f"Replacing sky using library ({self.sky_library.count} options)...")
            result = self._blend_sky_library(image, sky_mask, sky_index)
        else:
            log.info("Replacing sky using SDXL inpainting...")
            method = "sdxl"
            result = self._inpaint_sky_sdxl(image, sky_mask)
        
        # Save result
        if output_path is None:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(ENHANCED_OUTPUT_FOLDER, f"{basename}_sky_replaced.jpg")
        
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Save mask for debugging
        mask_path = output_path.replace('.jpg', '_sky_mask.png')
        cv2.imwrite(mask_path, sky_mask)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        log.info(f"✅ Sky replaced in {elapsed:.1f}s → {output_path}")
        
        return {
            "status": "success",
            "method": method,
            "input_path": image_path,
            "output_path": output_path,
            "mask_path": mask_path,
            "sky_ratio": round(sky_ratio, 3),
            "dimensions": {"width": width, "height": height},
            "processing_time_s": round(elapsed, 1)
        }
    
    def get_info(self) -> dict:
        """Get service info"""
        self.sky_library.load()
        return {
            "models_loaded": self._models_loaded,
            "device": self.device,
            "sky_library": {
                "available": self.sky_library.available,
                "count": self.sky_library.count,
                "skies": self.sky_library.list_skies()
            },
            "config": {
                "min_sky_ratio": self.config.min_sky_ratio,
                "max_sky_ratio": self.config.max_sky_ratio,
                "feather_amount": self.config.feather_amount,
                "use_library": self.config.use_library
            }
        }


# Global singleton
sky_replacer = SkyReplacer()
