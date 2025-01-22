# MARIDA Multi-Model Analysis

This repository is a fork from the **MARIDA** dataset project, aiming to demonstrate various machine learning techniques for marine debris detection and classification. The dataset (and original codebase) can be found at the official [MARIDA GitHub repository](https://github.com/marine-debris). We extend that work here by showing:

1. **Random Forest** classification using spectral signatures.
2. **U-Net** for pixel-level semantic segmentation.
3. **YOLO** (Ultralytics) for object detection.

## Repository Structure

- **notebooks/**  
  Contains the main Jupyter Notebook(s) illustrating the above models in detail.  

- **data/**  
  Holds the MARIDA dataset files (HDF5, `.tif` patches, shapefiles, etc.).  
  (**Ignored** from version control via `.gitignore` for size/privacy reasons.)

- **.gitignore**  
  Configuration for ignoring large data folders, logs, or any sensitive files.


