# Real Estate Photography - AI Image Processing Pipeline

> Production-Grade Automated Image Editing System  
> Documentation compiled: January 2, 2026

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Workflow](#system-workflow)
3. [Requirements](#requirements)
4. [Tech Stack](#tech-stack)
5. [AI Models Explained](#ai-models-explained)
6. [Hardware Analysis](#hardware-analysis)
7. [Image Processing Details](#image-processing-details)
8. [AWS Deployment](#aws-deployment)
9. [Cost Estimates](#cost-estimates)

---

## Project Overview

### Goal

Build a **high-throughput, automated image-editing pipeline** for real estate photography that transforms RAW HDR property photos into market-ready, color-corrected, and exposure-balanced images—with optional sky replacement for exteriors—while preserving:

- ✅ Strict scene fidelity
- ✅ Original resolution
- ✅ Batch consistency

### Core Problem Being Solved

| Challenge | Solution |
|-----------|----------|
| Manual editing is slow & expensive | Fully automated AI pipeline |
| 10,000+ photos/month volume | Scalable GPU processing |
| Inconsistent editing styles | Standardized AI models + QC |
| Poor sky in exterior shots | Automatic sky replacement |
| Quality degradation from AI | ControlNets + low denoise preservation |

---

## System Workflow

### Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                       IMAGE PROCESSING FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌──────────────┐                                             │
│    │ Image Ingest │                                             │
│    │ HDR Merged   │                                             │
│    └──────┬───────┘                                             │
│           ▼                                                      │
│    ┌──────────────────┐                                         │
│    │ Normalize & Clamp │                                        │
│    │ sRGB Conversion   │                                        │
│    └──────┬────────────┘                                        │
│           ▼                                                      │
│    ┌──────────────────┐                                         │
│    │ Interior or      │                                         │
│    │ Exterior?        │──────────────────────┐                  │
│    └──────┬───────────┘                      │                  │
│           │                                   │                  │
│    ┌──────▼───────────────────────────┐      │                  │
│    │ Base Image Correction            │◄─────┘                  │
│    │ SDXL Lightning img2img           │                         │
│    │ Low Denoise                      │                         │
│    └──────┬───────────────────────────┘                         │
│           ▼                                                      │
│    ┌──────────────────┐                                         │
│    │ ControlNet Tile  │ ← Texture & Color Lock                  │
│    └──────┬───────────┘                                         │
│           ▼                                                      │
│    ┌──────────────────────┐                                     │
│    │ ControlNet Canny/    │ ← Geometry Lock                     │
│    │ Lineart              │                                     │
│    └──────┬───────────────┘                                     │
│           │                                                      │
│     ┌─────┴─────┐                                               │
│     ▼           ▼                                               │
│ [INTERIOR]  [EXTERIOR]                                          │
│     │           │                                               │
│     ▼           ▼                                               │
│ ┌─────────┐  ┌─────────────────┐                               │
│ │Interior │  │Sky Segmentation │                               │
│ │LoRA     │  │(U²-Net/SAM2)    │                               │
│ │Low Wt   │  └────────┬────────┘                               │
│ └────┬────┘           ▼                                         │
│      │         ┌─────────────────┐                              │
│      │         │Sky Mask         │                              │
│      │         │Generation       │                              │
│      │         └────────┬────────┘                              │
│      │                  ▼                                        │
│      │         ┌─────────────────┐                              │
│      │         │SDXL Inpainting  │                              │
│      │         │(Sky Only)       │                              │
│      │         └────────┬────────┘                              │
│      │                  ▼                                        │
│      │         ┌─────────────────┐                              │
│      │         │Histogram Match  │                              │
│      │         │& Composite      │                              │
│      │         └────────┬────────┘                              │
│      │                  │                                        │
│      └────────┬─────────┘                                       │
│               ▼                                                  │
│    ┌──────────────────────────┐                                 │
│    │High-Resolution Tiled     │                                 │
│    │Reconstruction            │                                 │
│    └──────────┬───────────────┘                                 │
│               ▼                                                  │
│    ┌──────────────────────────┐                                 │
│    │Automated QC Checks       │                                 │
│    │SSIM · Color Delta · Edge │                                 │
│    └──────────┬───────────────┘                                 │
│               │                                                  │
│        ┌──────┴──────┐                                          │
│        ▼             ▼                                          │
│    [PASS]        [FAIL]                                         │
│        │             │                                          │
│        ▼             ▼                                          │
│ ┌────────────┐ ┌────────────┐                                  │
│ │Final Export│ │Reprocess   │──────────┐                       │
│ │Full Res    │ │Lower Denoise│         │                       │
│ └────────────┘ └────────────┘         │                        │
│                      ▲                 │                        │
│                      └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Process

1. **Ingest** → Single HDR image OR bracket set (auto-merge available)
2. **HDR Merge** (optional) → Built-in OpenCV Exposure Fusion if brackets provided
3. **Normalize** → Convert to sRGB, clamp values
4. **Classify** → AI determines if interior or exterior
5. **Base Correction** → SDXL Lightning with low denoise for subtle fixes
6. **ControlNets** → Lock textures (Tile) and geometry (Canny/Lineart)
7. **Branch Processing:**
   - *Interior:* Apply custom-trained LoRA for lighting/style + **Window Pull** (recover blown-out windows)
   - *Exterior:* Segment sky → Generate mask → Inpaint new sky → Histogram match & composite
8. **High-Res Tiled Reconstruction** → Stitch tiles back for full resolution
9. **Quality Control (QC)** → SSIM, color delta, edge validation checks
10. **Error Reporting** → Log failures, notifications, retry queue
11. **Export or Retry** → Pass = final export; Fail = reprocess with lower denoise

---

## Requirements

### Business Requirements

| Requirement | Details |
|-------------|---------|
| **Volume** | Handle **10,000+ photos per month** |
| **Photo Types** | Indoor (living rooms, kitchens) AND outdoor (house exteriors) |
| **Realism** | Keep photos looking **real** — no fake or AI-artifact-looking results |
| **Quality Preservation** | Maintain **original resolution** (no blurry or shrunk images) |
| **Consistency** | All photos must have a **consistent style** across batches |
| **Input Methods** | Accept photos via **folder drop** or **web service/API** |

### Technical Requirements

#### Input & HDR Processing
- Accept **pre-merged HDR images** (if user has existing workflow)
- **Built-in HDR merging** using OpenCV Exposure Fusion (no external software needed)
- Support bracket sets (3-7 exposures) for auto-merge
- Automatically classify photos as **interior** or **exterior**
- Normalize and convert to sRGB

#### Image Processing
- Fix **brightness, exposure, and dark areas**
- Correct **colors** to look natural
- Preserve **geometry and object positions** exactly
- Remove **harsh shadows** and dark corners (interiors)

#### Window Pull (Interiors)
- Detect **blown-out windows** (overexposed white areas)
- Segment window regions using SAM2
- **Inpaint natural outdoor views** through windows (trees, sky, etc.)
- Blend edges for seamless integration

#### Sky Replacement (Exteriors)
- Detect and segment the sky region
- Replace dull/cloudy skies with **clear blue skies**
- Blend new sky to look **natural** (histogram matching)

#### Resolution Handling
- Support high-resolution photos (**8183×5455 pixels / 45MP**)
- Process in **tiles** to avoid memory issues
- Reconstruct full resolution with **seam correction**

#### Quality Control
- Automated QC checks:
  - **SSIM** (structural similarity)
  - **Color Delta** (color accuracy)
  - **Edge Validation** (geometry preservation)
- Auto-retry failed images with lower denoise settings

#### Error Reporting
- **Per-image error logs** with failure reasons
- **Failed image quarantine** folder for review
- **Notifications** (Slack/Email/Webhook) for batch failures
- **Dashboard** with failure analytics and retry history
- Error codes for specific failures (sky detection, window detection, QC fail, etc.)

#### Scalability & Reliability
- Run on **multiple computers/GPUs** simultaneously
- Continue working if one part fails (**fault tolerance**)
- Preserve all **original metadata**

### Anti-Requirements (What to Avoid)

- ❌ No **hallucinations** or AI artifacts
- ❌ No **color drift** from original scene
- ❌ No **resolution loss** or quality degradation
- ❌ No **creative reinterpretation** — diffusion is used as a *constrained operator*

---

## Tech Stack

| Component | Technology / Model | Purpose |
|-----------|-------------------|---------|
| HDR Merge | OpenCV Exposure Fusion | Merge bracket exposures (built-in) |
| Base Model | SDXL Lightning | Fast image correction (4 steps) |
| ControlNet | Tile + Canny/Lineart | Preserve textures & geometry |
| Sky Segmentation | U²-Net / SAM2 (fine-tuned) | Detect sky regions |
| Window Detection | SAM2 | Detect blown-out windows (interiors) |
| Classifier | CLIP / Lightweight CNN | Interior/Exterior detection |
| Inpainting | SDXL Inpainting | Sky & window replacement |
| LoRA | Custom-trained for interiors | Professional look enhancement |
| Backend | Python, Flask, Redis (queue) | Job processing |
| Error Reporting | Logging + Notifications | Track failures, alerts, retry |
| GPU Management | CUDA, TensorRT (optional) | Optimization |

---

## AI Models Explained

### 1. SDXL Lightning (Base Model)

**What it is:** A fast version of Stable Diffusion XL (image generation AI)

**What it does in this project:**
- **NOT generating new images** — just making subtle fixes
- Corrects exposure, lighting, color balance
- Uses "img2img" mode with **low denoise (0.2-0.4)** = keeps 60-80% of original
- "Lightning" = only 4 steps instead of 20-50 = much faster

**Example improvements:**
| Before | After SDXL |
|--------|------------|
| Dark corners | Evenly lit |
| Yellow color cast | Neutral colors |
| Harsh shadows | Softer shadows |

---

### 2. ControlNet Tile

**What it is:** An add-on that forces SDXL to preserve textures and colors

**What it does:**
- Locks down **surface textures** (wood grain, carpet, tiles)
- Preserves **color palette** of the original
- Prevents AI from changing materials (no wood → marble mistakes)

**Why it's critical:** Without this, SDXL might "improve" a hardwood floor into something completely different!

---

### 3. ControlNet Canny / Lineart

**What it is:** An add-on that forces SDXL to preserve edges and geometry

**What it does:**
- Extracts **all edges** from the image (walls, furniture, windows)
- Forces SDXL to keep objects in **exact same position**
- Prevents geometry distortion (no wobbly walls!)

| Canny | Lineart |
|-------|---------|
| Detects ALL edges | Detects main structural lines |
| More precise | More forgiving |
| Better for architecture | Better for soft objects |

---

### 4. U²-Net / SAM2 (Sky Segmentation)

**What it is:** AI that identifies and separates the sky from the rest of the image

**U²-Net:**
- Lightweight (~50MB)
- Fast (~0.5 sec)
- Great for clear sky boundaries

**SAM2 (Segment Anything Model 2):**
- More accurate for complex scenes
- Handles trees, wires, chimneys better
- Slightly slower

**Output:** A precise binary mask (white = sky, black = building)

---

### 5. CLIP (Classifier)

**What it is:** AI that understands what's in an image (by OpenAI)

**What it does in this project:**
- Decides: **Interior** or **Exterior**?
- Can also detect: kitchen, bedroom, living room, bathroom, etc.
- Very fast (~100ms)
- No training needed — works out of the box

---

### 6. SDXL Inpainting

**What it is:** SDXL but specifically for filling in masked areas

**What it does:**
- Only regenerates the **masked area** (sky)
- Keeps everything else **untouched**
- Generates realistic blue skies with clouds
- Blends edges naturally

---

### 7. Custom Interior LoRA

**What it is:** A small fine-tuned model trained on professional interior photos

**What it does:**
- Trained on ~500-1000 professional real estate interior photos
- Adds that "Zillow/Redfin" professional look
- Applied with **low weight (0.3-0.4)** = subtle effect
- Improves: warm lighting, balanced exposure, appealing colors

| LoRA Weight | Effect |
|-------------|--------|
| 0.1-0.3 | Very subtle, natural |
| 0.4-0.5 | Noticeable improvement |
| 0.6+ | Risk of "fake" look ❌ |

---

### Model Summary Table

| Model | Purpose | VRAM | Speed |
|-------|---------|------|-------|
| **SDXL Lightning** | Color/exposure fixes | ~5 GB | Fast (4 steps) |
| **ControlNet Tile** | Lock textures | ~1.5 GB | — |
| **ControlNet Canny** | Lock geometry | ~1.5 GB | — |
| **U²-Net** | Find sky | ~200 MB | Very fast |
| **SAM2** | Find sky (better) | ~1 GB | Fast |
| **CLIP** | Interior/Exterior? | ~400 MB | Very fast |
| **SDXL Inpainting** | Replace sky | ~5 GB | Fast |
| **Interior LoRA** | Pro photo look | ~200 MB | — |

---

## Hardware Analysis

### Local Development: RTX 4060 (8GB VRAM)

#### Feasibility

| Component | VRAM Usage | Status |
|-----------|------------|--------|
| **SDXL Lightning** (4-step) | ~5-6 GB | ✅ Works with optimizations |
| **ControlNet Tile** | ~1-2 GB | ✅ Fits |
| **ControlNet Canny/Lineart** | ~1-2 GB | ⚠️ Load one at a time |
| **U²-Net (sky segmentation)** | ~200 MB | ✅ Very lightweight |
| **CLIP classifier** | ~400 MB | ✅ Lightweight |
| **Custom LoRA** | ~50-200 MB | ✅ Minimal overhead |

#### Required Optimizations for 8GB VRAM

1. Use **FP16** (half precision) for all models
2. Enable **xformers** or PyTorch 2.0 SDPA for memory-efficient attention
3. **Sequential ControlNet loading** (not parallel)
4. **Tiled VAE decoding** (already planned)
5. **Model offloading** to CPU between stages
6. Use **SDXL Lightning** (4-step = faster, less VRAM)
7. Optional: **TensorRT** optimization for 30-50% speedup

#### Expected Performance (RTX 4060)

| Task | Estimated Time |
|------|----------------|
| Classify image (CLIP) | ~0.1 sec |
| Base correction (SDXL Lightning 4-step, 1024px tile) | ~2-4 sec/tile |
| Sky segmentation | ~0.5 sec |
| Sky inpainting | ~3-5 sec |
| Full image (8183×5455, ~96 tiles at 768px) | ~3-4 min |

**Monthly capacity:** ~10,000-15,000 images/month (running 24/7)

---

## Image Processing Details

### Input Image Specifications

| Attribute | Value |
|-----------|-------|
| **Resolution** | 8183 × 5455 pixels (~45 MP) |
| **Format** | JPG, PNG, TIFF, or RAW (HDR merged) |
| **Color Space** | sRGB (will convert if needed) |
| **Bit Depth** | 8-bit or 16-bit |

### Tiling Strategy

Using **768×768 tiles** with **64px overlap** (safe for 8GB VRAM):

```
Image:     8183 × 5455 pixels
Tile size: 768 × 768 (with 64px overlap)

Horizontal tiles: ceil(8183 / 704) = 12 tiles
Vertical tiles:   ceil(5455 / 704) = 8 tiles
────────────────────────────────────────────
Total tiles:      12 × 8 = ~96 tiles per image
```

### Memory Usage Estimate

| Component | RAM (System) | VRAM (GPU) |
|-----------|--------------|------------|
| Load full 8183×5455 image | ~135 MB | — |
| Working buffers | ~500 MB | — |
| SDXL model | — | ~5 GB |
| ControlNet (1 at a time) | — | ~1.5 GB |
| Tile processing | — | ~1 GB |
| **Total** | **~2 GB** | **~7.5 GB** |

**System RAM needed:** At least **16 GB** recommended (32 GB ideal for batch queuing)

---

## AWS Deployment

### Recommended Instances

| Instance | GPU | VRAM | vCPUs | $/hr (On-Demand) | $/hr (Spot) | Best For |
|----------|-----|------|-------|------------------|-------------|----------|
| **g5.xlarge** | A10G | 24 GB | 4 | $1.00 | ~$0.35 | ✅ Best value |
| **g5.2xlarge** | A10G | 24 GB | 8 | $1.21 | ~$0.45 | More CPU |
| **g6.xlarge** | L4 | 24 GB | 4 | $0.80 | ~$0.30 | Newest |
| **g5.12xlarge** | 4× A10G | 96 GB | 48 | $5.67 | ~$2.00 | High throughput |

### Primary Recommendation: g5.xlarge

```
Instance Type:    g5.xlarge
AMI:              Deep Learning AMI GPU PyTorch 2.0 (Ubuntu 20.04)
Spot:             Yes (with fallback to on-demand)
Storage:          200 GB gp3 SSD
Region:           us-east-1 (cheapest) or ap-south-1 (Mumbai)
```

**Specs:**
- GPU: NVIDIA A10G (24 GB VRAM)
- Tensor Cores: Yes (Ampere architecture)
- CUDA Cores: 9,216
- Max tile size: 1280×1280 (comfortable)
- Images/hour: ~50

### AWS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         AWS Cloud                           │
│                                                             │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │   S3 Input  │────▶│   SQS Queue  │────▶│  g5.xlarge  │  │
│  │   Bucket    │     │  (Job Queue) │     │  (Spot ASG) │  │
│  └─────────────┘     └──────────────┘     └──────┬──────┘  │
│                                                   │         │
│                                                   ▼         │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │  S3 Output  │◀────│   Lambda     │◀────│   Results   │  │
│  │   Bucket    │     │  (Metadata)  │     │             │  │
│  └─────────────┘     └──────────────┘     └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### AWS Services Required

| Service | Purpose |
|---------|---------|
| **EC2 g5/g6** | GPU processing |
| **S3** | Image storage (input/output) |
| **SQS** | Job queue |
| **Auto Scaling Group** | Spin up/down based on queue |
| **Spot Instances** | 70% cost savings |
| **ECR** | Docker container registry |
| **CloudWatch** | Monitoring & logs |

### Best Regions for Spot Pricing

| Region | g5.xlarge Spot Price | Availability |
|--------|---------------------|--------------|
| us-east-1 (N. Virginia) | ~$0.30-0.40 | ✅ High |
| us-west-2 (Oregon) | ~$0.35-0.45 | ✅ High |
| ap-south-1 (Mumbai) | ~$0.25-0.35 | ⚠️ Medium |
| eu-west-1 (Ireland) | ~$0.35-0.45 | ✅ High |

---

## Cost Estimates

### Local RTX 4060 (Development)

| Item | Cost |
|------|------|
| Hardware | Already owned |
| Electricity (~200W × 24/7) | ~$20-30/month |
| **Total** | **~$25/month** |

### AWS g5.xlarge (Production)

Using Spot pricing for 10K images/month:

```
Images:             10,000
Time per image:     ~75 sec (average)
Total GPU hours:    ~208 hours
Spot price:         $0.35/hr
────────────────────────────────
Monthly cost:       ~$73/month
```

| Pricing Model | Monthly Cost (10K images) |
|---------------|---------------------------|
| **Spot Instance** | ~$70-100 |
| **On-Demand** | ~$200-250 |
| **Reserved (1yr)** | ~$150/month |

### Scaling Costs

| Scale | AWS Setup | Cost/Month |
|-------|-----------|------------|
| **10K images** | 1× g5.xlarge (Spot) | ~$75 |
| **30K images** | 2× g5.xlarge (Spot ASG) | ~$200 |
| **50K images** | 1× g5.12xlarge (4 GPUs) | ~$350 |
| **100K+ images** | Multiple g5.12xlarge | ~$700+ |

---

## Summary

### Hardware Recommendations by Scale

| Your Needs | Best Choice | Why |
|------------|-------------|-----|
| **Development** | RTX 4060 (owned) | Free, you have it |
| **10K-20K/month** | AWS g5.xlarge Spot | ~$75/month, no upfront |
| **20K-50K/month** | AWS g5.2xlarge or buy RTX 4090 | Better throughput |
| **50K+/month** | AWS g5.12xlarge or 2× 4090 | Professional scale |

### Key Metrics

| Metric | Value |
|--------|-------|
| Input Resolution | 8183 × 5455 (45 MP) |
| Output Resolution | Same as input |
| Tiles per image | ~96 (at 768px) |
| Time per image (RTX 4060) | ~3-4 minutes |
| Time per image (AWS g5) | ~60-90 seconds |
| Monthly capacity (g5.xlarge) | ~10,000-15,000 |

---

## Next Steps

1. [ ] Set up local development environment (RTX 4060)
2. [ ] Download and configure required models
3. [ ] Build core processing pipeline
4. [ ] Test with sample images
5. [ ] Optimize for 8GB VRAM
6. [ ] Deploy to AWS for production

---

*Document Version: 1.0*  
*Last Updated: January 2, 2026*
