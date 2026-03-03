# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pypylon is an industrial vision system for Basler cameras that captures images, performs image processing (thresholding, connected component analysis, morphological operations), and exports contours to DXF format for use with RoboDK or similar robotics software.

## Commands

### Run the API server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Run the main camera capture loop
```bash
python Pypylon/main.py
```

### Run YOLO inference
```bash
python Pypylon/CNN_main.py
```

## Architecture

### Core Modules

- **main.py**: Core image processing library containing:
  - `BaslerCamera`: Wrapper class for pypylon camera operations (connect by IP or serial, grab single frames or continuous loop)
  - Image processing functions: `threhold()`, `connectedComponents()`, `connectedComponentsOnly()`
  - DXF export: `contours_to_dxf_as_block()` - converts OpenCV contours to DXF BLOCK entities centered at their centroid

- **api.py**: FastAPI REST interface exposing image processing as stateless endpoints. Images are passed as base64-encoded PNGs. Endpoints:
  - `/camera/connect`, `/camera/capture` - camera operations
  - `/image/threshold`, `/image/connected-components`, `/image/erode`, `/image/find-circle` - image processing
  - `/export/dxf` - DXF file generation

- **CNN_main.py**: YOLO-based object detection for wheel rim classification. Contains functions for training (`train_data()`) and inference (`inference_data()`).

- **pylon_main.py**: Minimal example of continuous camera capture with pypylon.

### Key Dependencies

- `pypylon`: Basler camera SDK bindings
- `opencv-python`: Image processing
- `ezdxf`: DXF file generation (uses R2007 format with millimeter units)
- `ultralytics`: YOLO model for object detection
- `fastapi` + `uvicorn`: REST API

### Data Flow

1. Camera captures BGR image via `BaslerCamera`
2. Thresholding converts to binary mask
3. Connected component analysis filters by area and extracts contours
4. Optional morphological erosion refines contours
5. Contours are exported to DXF as a single BLOCK, centered at the centroid

### DXF Export Details

The `contours_to_dxf_as_block()` function:
- Flips Y-axis (image to DXF coordinate conversion)
- Applies Douglas-Peucker approximation (`eps_ratio` controls simplification)
- Creates all contours as a single BLOCK for RoboDK compatibility
- Scales coordinates (default 0.15 for pixel-to-mm conversion)
