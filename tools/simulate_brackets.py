import os
import sys
import cv2
import numpy as np
from pathlib import Path


def simulate_brackets(
    image_path: str,
    output_dir: str = None,
    num_brackets: int = 3,
    ev_step: float = 2.0
) -> list:
    """
    Create simulated exposure brackets from a single image
    
    Args:
        image_path: Path to source image
        num_brackets: Number of brackets to create (3, 5, or 7)
        ev_step: Exposure value step between brackets
        output_dir: Output directory (default: same as input)
    
    Returns:
        List of created file paths
    """
    if num_brackets not in [3, 5, 7]:
        raise ValueError("num_brackets must be 3, 5, or 7")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_float = img.astype(np.float32) / 255.0
    
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = Path(image_path).stem
    
    half = num_brackets // 2
    offsets = [i * (ev_step / half) for i in range(-half, half + 1)]
    
    output_paths = []
    
    for i, ev in enumerate(offsets):
        multiplier = 2 ** ev
        
        adjusted = np.clip(img_float * multiplier, 0, 1)
        
        adjusted_8bit = (adjusted * 255).astype(np.uint8)
        
        if ev < 0:
            ev_label = f"m{abs(ev):.0f}"  
        elif ev > 0:
            ev_label = f"p{ev:.0f}" 
        else:
            ev_label = "0"
        
        output_name = f"{base_name}_bracket_{i+1}_{ev_label}EV.jpg"
        output_path = os.path.join(output_dir, output_name)
        
        cv2.imwrite(output_path, adjusted_8bit, [cv2.IMWRITE_JPEG_QUALITY, 95])
        output_paths.append(output_path)
        print(f"Created: {output_name} (EV: {ev:+.1f})")
    
    print(f"\n✅ Created {num_brackets} brackets in: {output_dir}")
    return output_paths


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulate_brackets.py <image_path> [num_brackets] [output_dir]")
        print("  num_brackets: 3, 5, or 7 (default: 3)")
        print("  output_dir: where to save brackets (default: same as input)")
        print("\nExample:")
        print("  python simulate_brackets.py photo.jpg 3 ./test_brackets/")
        sys.exit(1)
    
    image_path = sys.argv[1]
    num_brackets = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    simulate_brackets(image_path, output_dir, num_brackets)
