# Padel Shot Detection System

A computer vision system that detects and classifies Padel shots from video footage. The system uses YOLOv8 for player detection with multi-object tracking and MediaPipe for pose estimation to classify shot types.

## Overview

This project processes Padel video footage to:
- **Detect players** in each frame using YOLOv8 nano model
- **Track players** across frames with consistent ID assignment
- **Extract pose landmarks** using MediaPipe Pose for body keypoint detection
- **Classify shots** into categories: smash, forehand, backhand, or unknown
- **Generate annotated video** with bounding boxes and shot type labels
- **Produce analytics** including shot statistics per player

## Features

- **Multi-Object Tracking**: Maintains consistent player IDs across video frames
- **Shot Classification**: Classifies Padel shots based on arm and wrist position relative to body
  - **Smash**: Right wrist positioned high above shoulder
  - **Forehand**: Right wrist on the right side of the body center
  - **Backhand**: Right wrist crosses to the left side of body center
  - **Unknown**: When landmarks cannot be reliably detected
- **GPU Acceleration**: CUDA support for faster video processing
- **Real-time Visualization**: Annotates video with player bounding boxes and shot classifications

## Project Structure

```
.
├── main.py                      # Main video processing pipeline
├── src/
│   ├── detector.py             # YOLOv8-based player detection and tracking
│   ├── shot_classifier.py       # MediaPipe-based shot classification
│   └── analytics.py            # Analytics and statistics generation
├── notebook/
│   └── Exploration.ipynb        # Data exploration and analysis notebook
├── data/                        # Input video directory
│   └── input.mp4               # Source Padel video
├── results/                     # Output directory
│   ├── results.json            # Detection results in JSON format
│   └── analytics.json          # Shot statistics and analytics
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Requirements & Setup

### System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for real-time processing)
- ~4GB GPU memory (for YOLOv8n)

### Installation

1. **Clone/Setup the project**:
   ```bash
   cd cv task
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Key Dependencies

- **ultralytics**: YOLOv8 object detection and tracking
- **opencv-python**: Video I/O and frame processing
- **torch, torchvision**: PyTorch for GPU acceleration
- **mediapipe**: Pose landmark detection
- **pandas**: Data manipulation and analysis
- **matplotlib**: Visualization and analytics
- **jupyter**: Interactive notebook environment

## Usage

### Running the Detection Pipeline

```bash
python main.py
```

This will:
1. Load the video from `data/input.mp4`
2. Process each frame (sampling every 3rd frame for efficiency)
3. Detect the top 2 most prominent players per frame
4. Classify their shot types using pose landmarks
5. Generate an annotated output video `output.mp4`
6. Log detection results in `results.json`

### Processing Details

- **Frame Sampling**: Every 3rd frame is processed to reduce computation
- **Player Selection**: Top 2 players by bounding box area are analyzed per frame
- **GPU Processing**: Player detection and pose estimation run on GPU when available
- **Progress Tracking**: Frame processing status is printed every 150 frames

### Generating Analytics

```bash
python src/analytics.py
```

This script processes results and generates:
- Overall shot type counts
- Shot counts per player
- Shots per minute per player
- Analytics saved to `results/analytics.json`

### Exploring Data

Use the Jupyter notebook for interactive data exploration:
```bash
cd notebook

jupyter notebook
```

## Models

- **YOLOv8n** (`yolov8n.pt`): Nano model for fast person detection and tracking
- **MediaPipe Pose**: Pre-trained pose estimation model (loaded automatically)

## Output

### Video Output
- `output.mp4`: Annotated video with player bounding boxes and shot type labels

### Data Output
- `results/results.json`: Frame-by-frame detection data including:
  - Frame number and timestamp
  - Player ID
  - Detected shot type
  
- `results/analytics.json`: Aggregated statistics:
  - Overall shot type distribution
  - Shots per player by type
  - Shots per minute per player

## Performance Considerations

- **Frame Sampling**: Processing every 3rd frame reduces computation by ~67%
- **GPU Usage**: Significantly faster inference with CUDA-capable GPU
- **Model Size**: YOLOv8n is optimized for speed with minimal accuracy loss compared to larger models
- **Processing Time**: ~30-60% of real-time speed on GPU, depends on video resolution

## Notes

- The system is optimized for tracking the top 2 players (highest detection confidence)
- Shot classification relies on reliable pose landmark detection; occluded or partially visible players may return "unknown"
- Video codec support depends on FFmpeg installation; MP4 is recommended
