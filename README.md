# Pathology Image Processing Pipeline

This project provides a comprehensive pipeline for processing pathology slide images (SVS files) to detect tissue regions, extract patches, and classify background/foreground areas.

## Overview

The pipeline consists of three main stages:

1. **Tissue Detection** (`1.find_tissue.py`) - Automatically detects tissue regions in SVS files
2. **Patch Extraction** (`2.crop_patch_20x.py`) - Extracts 20x magnification patches from detected tissues
3. **Background/Foreground Classification** (`3.find_bg_fg.py`) - Classifies patches as background or foreground

## Features

- **Automatic tissue detection** using HSV color space and saturation analysis
- **Multi-resolution support** for SVS files
- **Patch extraction** at 20x magnification with 512x512 pixel size
- **Background/foreground classification** based on image statistics
- **Comprehensive output** including images, masks, and Excel files with metadata
- **Debug visualization** for each processing stage
- **Command-line interface** with flexible argument options

## Requirements

- Python 3.7+
- See `requirements.txt` for detailed dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pathskin_share
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Stage 1: Tissue Detection

Detects tissue regions in SVS files and generates bounding boxes.

```bash
python 1.find_tissue.py --input_dir /path/to/svs/files --output_dir /path/to/output
```

**Arguments:**
- `--input_dir`: Input directory containing SVS files (required)
- `--output_dir`: Output directory for results (required)
- `--recursive`: Search for SVS files recursively in subdirectories (default: True)

**Example:**
```bash
python 1.find_tissue.py \
    --input_dir /data/pathology/svs_files \
    --output_dir /output/tissue_detection \
    --recursive
```

**Input:**
- SVS files in the specified input directory
- Annotation files (.qpdata) for metadata

**Output:**
- Processed images with bounding boxes
- `bboxes.xlsx` - Bounding box coordinates
- `infos.xlsx` - File metadata and magnification information
- Debug images (original, saturation, binary, etc.)

### Stage 2: Patch Extraction

Extracts 20x magnification patches from detected tissue regions.

```bash
python 2.crop_patch_20x.py --input_dir /path/to/svs/files --bbox_excel /path/to/bboxes.xlsx --output_dir /path/to/output
```

**Arguments:**
- `--input_dir`: Input directory containing SVS files (required)
- `--bbox_excel`: Path to bboxes.xlsx file from Stage 1 (required)
- `--output_dir`: Output directory for results (required)
- `--magnification`: Target magnification (default: 20)
- `--patch_size`: Patch size in pixels (default: 512)
- `--clear_masks`: Clear existing masks directory before processing

**Example:**
```bash
python 2.crop_patch_20x.py \
    --input_dir /data/pathology/svs_files \
    --bbox_excel /output/tissue_detection/bboxes.xlsx \
    --output_dir /output/patch_extraction \
    --magnification 20 \
    --patch_size 512
```

**Input:**
- SVS files
- `bboxes.xlsx` from Stage 1
- GeoJSON annotation files

**Output:**
- `patches/` - Individual patch images (JPG format)
- `masks/` - Patch masks (PNG format)
- `tissues/` - Visualization images with patch numbers
- `bboxes_patch_20x.xlsx` - Patch metadata and coordinates

### Stage 3: Background/Foreground Classification

Classifies patches as background or foreground based on image statistics.

```bash
python 3.find_bg_fg.py --base_dir /path/to/patch/data --output_dir /path/to/output
```

**Arguments:**
- `--base_dir`: Base directory containing patch data (output from Stage 2) (required)
- `--output_dir`: Output directory for results (required)
- `--file_id`: Specific file_id to process (if not specified, processes all)
- `--std_threshold`: Standard deviation threshold for background classification (default: 40.0)
- `--edge_threshold`: Edge intensity threshold for background classification (default: 15.0)
- `--output_suffix`: Suffix for output files (default: test)

**Example:**
```bash
python 3.find_bg_fg.py \
    --base_dir /output/patch_extraction \
    --output_dir /output/bg_fg_classification \
    --std_threshold 40.0 \
    --edge_threshold 15.0 \
    --output_suffix experiment1
```

**Input:**
- Patch images from Stage 2
- `bboxes_patch_20x.xlsx` from Stage 2
- Mask images

**Output:**
- Mask images with 'FG' labels on foreground patches
- `bboxes_bg_fg_{suffix}.xlsx` - Classification results with statistics

## Complete Pipeline Example

```bash
# Stage 1: Tissue Detection
python 1.find_tissue.py \
    --input_dir /data/pathology/svs_files \
    --output_dir /output/stage1_tissue_detection

# Stage 2: Patch Extraction
python 2.crop_patch_20x.py \
    --input_dir /data/pathology/svs_files \
    --bbox_excel /output/stage1_tissue_detection/bboxes.xlsx \
    --output_dir /output/stage2_patch_extraction

# Stage 3: Background/Foreground Classification
python 3.find_bg_fg.py \
    --base_dir /output/stage2_patch_extraction \
    --output_dir /output/stage3_bg_fg_classification \
    --output_suffix final
```

## Configuration

### Processing Parameters

Key parameters that can be adjusted via command line arguments:

**Stage 1 (Tissue Detection):**
- `--recursive`: Search subdirectories for SVS files

**Stage 2 (Patch Extraction):**
- `--magnification`: Target magnification (default: 20)
- `--patch_size`: Patch size in pixels (default: 512)
- `--clear_masks`: Clear existing masks before processing

**Stage 3 (Background/Foreground Classification):**
- `--std_threshold`: Standard deviation threshold (default: 40.0)
- `--edge_threshold`: Edge intensity threshold (default: 15.0)
- `--file_id`: Process specific file only
- `--output_suffix`: Custom suffix for output files

### Advanced Parameters

For more advanced customization, you can modify the following parameters in the code:

- **Tissue detection**: min_area=20000, aspect_ratio_threshold=0.1 (in `1.find_tissue.py`)
- **Patch overlap**: step_size = patch_size // 2 (50% overlap) (in `2.crop_patch_20x.py`)
- **Image quality**: JPEG quality settings for patch saving (in `2.crop_patch_20x.py`)

## File Structure

```
pathskin_share/
├── 1.find_tissue.py          # Tissue detection script
├── 2.crop_patch_20x.py       # Patch extraction script
├── 3.find_bg_fg.py           # Background/foreground classification
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── output/                   # Generated output (example structure)
    ├── stage1_tissue_detection/
    │   ├── images/
    │   ├── debug/
    │   ├── bboxes.xlsx
    │   └── infos.xlsx
    ├── stage2_patch_extraction/
    │   ├── patches/
    │   ├── masks/
    │   ├── tissues/
    │   └── bboxes_patch_20x.xlsx
    └── stage3_bg_fg_classification/
        ├── *_mask_with_fg.jpg
        └── bboxes_bg_fg_final.xlsx
```

## Algorithm Details

### Tissue Detection (Stage 1)

1. **Image preprocessing**: Remove dark areas and convert to HSV color space
2. **Saturation analysis**: Use saturation channel for tissue detection
3. **Binarization**: Apply threshold to create binary mask
4. **Morphological operations**: Remove noise and connect regions
5. **Bounding box generation**: Create and merge overlapping boxes

### Patch Extraction (Stage 2)

1. **Magnification detection**: Find appropriate resolution level for target magnification
2. **Coordinate transformation**: Convert between resolution levels
3. **Patch generation**: Extract patches with 50% overlap
4. **Mask creation**: Generate masks from GeoJSON annotations
5. **Visualization**: Create images showing patch locations

### Background/Foreground Classification (Stage 3)

1. **Statistics calculation**: Compute mean, standard deviation, and edge intensity
2. **Threshold-based classification**: Use statistical thresholds to determine background
3. **Visualization**: Mark foreground patches with 'FG' labels

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and Excel I/O
- **opencv-python**: Image processing
- **openslide-python**: SVS file reading
- **matplotlib**: Data visualization
- **tqdm**: Progress bars
- **openpyxl**: Excel file handling
- **Pillow**: Image processing support

## Troubleshooting

### Common Issues

1. **SVS file not found**: Ensure input directory path is correct
2. **GeoJSON file missing**: Check if annotation files exist for each SVS
3. **Memory errors**: Reduce batch size or use smaller images
4. **OpenSlide errors**: Verify SVS file integrity and OpenSlide installation
5. **Argument errors**: Check that all required arguments are provided

### Performance Tips

- Use SSD storage for faster I/O
- Adjust patch size based on available memory
- Process files in batches for large datasets
- Use multiprocessing for parallel processing (if implemented)
- Use `--file_id` to process specific files for testing

### Error Handling

All scripts include comprehensive error handling:
- Input validation for directories and files
- Graceful handling of missing files
- Detailed error messages and progress reporting
- Automatic creation of output directories

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```

## Contact

[Add contact information here]
