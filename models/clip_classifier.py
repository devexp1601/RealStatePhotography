"""
CLIP Model loader and classifier
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from config import CLIP_MODEL_NAME, CLASSIFICATION_LABELS


class CLIPClassifier:
    """Singleton CLIP model for image classification"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.processor = None
        self.device = None
        self._initialized = True
    
    def load(self):
        """Load the CLIP model (lazy loading)"""
        if self.model is not None:
            return
        
        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Use safetensors to avoid CVE-2025-32434
        self.model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME,
            use_safetensors=True
        )
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        print("CLIP model loaded successfully!")
    
    def classify(self, image_path: str) -> dict:
        """
        Classify an image as Interior or Exterior
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict with classification result and confidence scores
        """
        self.load()
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Process image and text
        inputs = self.processor(
            text=CLASSIFICATION_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Extract results
        probs = probs.cpu().numpy()[0]
        interior_score = float(probs[0])
        exterior_score = float(probs[1])
        
        # Determine classification
        if interior_score > exterior_score:
            classification = "interior"
            confidence = interior_score
        else:
            classification = "exterior"
            confidence = exterior_score
        
        return {
            "classification": classification,
            "confidence": round(confidence * 100, 2),
            "scores": {
                "interior": round(interior_score * 100, 2),
                "exterior": round(exterior_score * 100, 2)
            },
            "dimensions": {"width": width, "height": height}
        }
    
    @property
    def is_gpu_available(self) -> bool:
        return torch.cuda.is_available()


# Global singleton instance
classifier = CLIPClassifier()
