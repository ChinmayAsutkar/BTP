# Monogaze1

Monogaze1 is a unified pipeline for running multiple depth estimation and object detection models on images and videos. It enables easy experimentation, benchmarking, and visualization of results from state-of-the-art computer vision models.

## Features

- Run depth estimation on images and videos using various models.
- Perform object detection and segmentation.
- Aggregate and visualize results for comparison.
- Organized structure for models, assets, and results.

## File Structure

```plaintext
Monogaze1/
│── depthfm_inference.py
│── eval_image.py
│── eval_video.py
│── monovit_inference.py
│── run_all_models_image.py
│── run_image.py
│── run_video.py
│── requirements.txt
│── README.md
│
├── depth_inference/
│   │── depth_anything_v2.py
│   │── depthfm_inference.py
│   │── hrdepth_inference.py
│   │── marigold_inference.py
│   │── metric3d_inference.py
│   │── midas_inference.py
│   │── monodepth2.py
│   └── unidepth.py
│
├── models/
│   ├── depth_models/
│   │   ├── Depth_Anything_V2/
│   │   ├── DepthFM/
│   │   ├── HRDepth/
│   │   ├── Marigold/
│   │   ├── Metric3D/
│   │   ├── MiDaS/
│   │   ├── Mono_Depth_2/
│   │   ├── Unidepth/
│   │   └── ZoeDepth/
│   │
│   └── detection_models/
│       ├── bounding_box/
│       │   ├── yolov8n.pt
│       │   └── yolov8x.pt
│       └── segmentation/
│
├── assets/
│
├── test/
│   ├── image.jpg
│   └── video.mp4
│
├── results/
├── combined/
├── depth_anything/
├── depthfm/
├── hrdepth/
├── marigold/
├── metric3d/
├── midas/
├── monodepth2/
├── unidepth/
├── zoedepth/
│
└── utils/
    ├── constants.py
    ├── detection.py
    └── generic.py

## Included Models

### Depth Estimation Models
- Depth Anything V2
- DepthFM
- HRDepth
- Marigold
- Metric3D
- MiDaS
- MonoDepth2
- Unidepth
- ZoeDepth

### Detection Models
- YOLOv8 (bounding box detection)
- Segmentation models (e.g., SAM2)

## Usage

1. Place your input images/videos in `assets/test/`.
2. Run scripts such as `run_image.py` or `run_video.py` to process inputs.
3. Results will be saved in the `results/` directory for each model.
4. Compare and visualize outputs using the generated images and CSV files.

## Requirements

Install dependencies using: pip install -r requirements.txt