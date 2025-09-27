How to Run

Setup
- Place your input images in the data_dir
  (default: /Users/yuxuancai/cs180/cs180/Proj_1/code/img)
- Update the data_dir path in main.py to match your directory structure
- Ensure your images are in supported formats:
  .jpg, .jpeg, .tif, .tiff, .png

Running Modes
The script supports three execution modes:

1. Process a Single Image
   python main.py "<image_path>"
   Example:
   python main.py "/path/to/your/image/cathedral.jpg"

2. Process All Images in Directory
   python main.py all
   This will process all supported image files (jpg, tif, png) found in the data_dir.

3. Default Batch Processing
   python main.py
   Running without arguments defaults to processing all images in the data_dir (same as mode 2).

Output
- Processed images are saved as:
  result_<original_name>.jpg
- The console displays alignment offsets: G:(dx, dy), R:(dx, dy)

Example console output:
  Processing cathedral.jpg...
  Image size: (1024, 390), Using pyramid: False, Metric: ncc, Edges: True
  G:(2, 5), R:(3, 12)
  Saved result to result_cathedral.jpg