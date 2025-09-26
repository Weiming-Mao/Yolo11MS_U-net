1. `*.pt` and `*.pth` files are pretrained model weights.

2. `YOLO11MS_U-net.py` and `YOLO11MS_U-net_v2.py` are two programs that wrap the two-stage model into a GUI; `v2` uses SAM for adaptive grid-region extraction, while the other uses a standard region-extraction model for grid-cell extraction.

3. `generate_defect_image.py` and `generate_grids_image.py` are scripts for generating defect images and grid (mesh) images, respectively.

4. `cvmethods_coverage_quantification.py` and `cvmethods_defect_detection.py` are scripts that use classical computer-vision methods for coverage quantification and defect detection, respectively.
