# **Real Estate Photography**

Production-Grade Image Processing System

# **Project Overview**

We are building a high-throughput, automated image-editing pipeline for real estate photography. The system transforms RAW HDR property photos into market-ready, colour-corrected, and exposure-balanced images—with optional sky replacement for exteriors—while preserving strict scene fidelity, original resolution, and batch consistency.

**Core Requirements:**

* Handle over 10,000 photos every month  
* Work on both indoor photos (living rooms, kitchens) and outdoor photos (house exteriors)  
* Keep photos looking real — no fake or weird-looking results  
* Keep the original photo quality (no blurry or shrunk images)  
* Make all photos look consistent in style  
* Accept photos through folders or a web service

# **Main Features**

 **1\. Two Ways to Send Photos**

* Folder Watching: Drop photos in a folder, and the system picks them up automatically.  
* Web Upload: Send photos through the internet and get edited photos back.

 **2\. Smart Photo Sorting**

* The system automatically figures out if a photo is indoor or outdoor.  
* It uses this to decide how to edit each photo.

 **3\. Basic Photo Fixes**

* Fixes brightness and dark areas.  
* Corrects colors so they look natural.  
* Keeps everything in the photo exactly where it should be.

 **4\. Indoor Photo Improvements**

* Makes indoor lighting look better.  
* Removes harsh shadows or dark corners.

 **5\. Window Pull (Interiors)**

* Detects blown-out (overexposed) windows in indoor photos.  
* Recovers natural outdoor views through windows.  
* Blends the new view seamlessly with the interior.

 **6\. Sky Replacement for Outdoor Photos**

* Detects the sky in outdoor photos.  
* Replaces cloudy or dull skies with clear blue skies.  
* Blends the new sky so it looks natural.

 **7\. HDR Bracket Merging**

* Accepts multiple exposure brackets (3-7 photos).  
* Automatically merges them into a single HDR image.  
* No external software needed — built-in processing.

 **8\. Works with Large Photos**

* Can handle very high-resolution photos (like 6000×4000 pixels).  
* Processes them in pieces to avoid running out of memory.

 **9\. Automatic Quality Check**

* Checks if the edited photo looks good.  
* If something looks wrong, it tries to fix it again.

 **10\. Error Reporting**

* Logs all failures with detailed error reasons.  
* Sends notifications (Email/Slack) when batches fail.  
* Keeps failed images in a separate folder for review.

 **11\. Built to Handle Heavy Work**

* Can run on multiple computers at once.  
* Keeps working even if one part fails.  
* Saves all the original photo information.

# 

# **Tech Stack**

| Component | Technology / Model |
| :---- | :---- |
| Base Model | SDXL Lightning |
| ControlNet | Tile \+ Canny/Lineart |
| Sky Segmentation | U²-Net / SAM2 (fine-tuned) |
| Classifier | CLIP / Lightweight CNN |
| Inpainting | SDXL Inpainting |
| LoRA | Custom-trained for interiors |
| Backend | Python, Redis (queue) |
| GPU Management | CUDA, TensorRT (optional for optimization) |

# 

# 

# **System Workflow**

1\. Ingest → Normalize HDR → Classify (Interior/Exterior)

2\. Base Correction (Tile \+ ControlNet, low denoise)

3\. Conditioned Branch:

   • Interior → Apply LoRA (weight ≤0.4)

   • Exterior → Sky Mask → Inpaint → Composite

4\. Tile Merge → Seam Correction → Sharpen

5\. QC Check → Pass/Fail → Re-Process if Needed

6\. Deliver → Save \+ Metadata

# **Conclusion**

This pipeline represents a deterministic, production-ready photographic processing system that uses diffusion models as constrained operators—not creative tools. By combining:

* ControlNet for fidelity preservation  
* Tiled inference for resolution integrity  
* Segmentation-based sky replacement  
* Automated QC for consistency

We achieve a scalable, high-throughput solution that meets the strict requirements of professional real estate photography. The system is designed to be robust, repeatable, and resistant to the hallucinations and colour drifts that plague simpler AI-based editing approaches.

