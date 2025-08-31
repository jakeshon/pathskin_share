"""
This script is a program for automatically detecting and segmenting specific regions in pathology slide images (SVS files).

Main features:
1. Extract the lowest resolution image from SVS files
2. Detect regions of interest based on saturation using HSV color space
3. Generate bounding boxes for detected regions
4. Merge overlapping bounding boxes to determine final regions
5. Save results as images and Excel files

Processing steps:
- Image preprocessing: Remove dark areas, HSV conversion
- Region detection: Saturation channel analysis, binarization, morphological operations
- Post-processing: Bounding box generation and merging
- Result saving: Save images and bounding box coordinates

Output:
- Processed images (.png)
- Bounding box information (.xlsx)
- Debug intermediate results (original, saturation, binarization results, etc.)
"""

import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
import shutil
import argparse

def merge_overlapping_boxes(boxes):
    """
    Function to merge overlapping bounding boxes
    
    Args:
        boxes (list): List of bounding box coordinates in [x1, y1, x2, y2] format
        
    Returns:
        list: List of merged bounding box coordinates
    """
    if len(boxes) == 0:
        return []
    
    # Convert box coordinates to numpy array
    boxes = np.array(boxes)
    
    # Convert box coordinates to [x1, y1, x2, y2] format
    boxes_xyxy = boxes.copy()
    
    # Sort boxes by area (larger boxes first)
    area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    idxs = np.argsort(area)[::-1]
    
    # Store selected box indices
    pick = []
    
    while len(idxs) > 0:
        # Select the first box
        i = idxs[0]
        pick.append(i)
        
        # Check overlap with remaining boxes
        xx1 = np.maximum(boxes_xyxy[i, 0], boxes_xyxy[idxs[1:], 0])
        yy1 = np.maximum(boxes_xyxy[i, 1], boxes_xyxy[idxs[1:], 1])
        xx2 = np.minimum(boxes_xyxy[i, 2], boxes_xyxy[idxs[1:], 2])
        yy2 = np.minimum(boxes_xyxy[i, 3], boxes_xyxy[idxs[1:], 3])
        
        # Calculate width and height of overlapping area
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # Find indices of overlapping boxes (if any overlap)
        overlapping_indices = np.where((w > 0) & (h > 0))[0] + 1
        
        if len(overlapping_indices) > 0:
            # Get coordinates of all boxes overlapping with current box
            overlapping_boxes = boxes_xyxy[np.concatenate(([i], idxs[overlapping_indices]))]
            
            # Calculate new box coordinates (using min/max coordinates)
            new_box = [
                np.min(overlapping_boxes[:, 0]),  # xmin
                np.min(overlapping_boxes[:, 1]),  # ymin
                np.max(overlapping_boxes[:, 2]),  # xmax
                np.max(overlapping_boxes[:, 3])   # ymax
            ]
            
            # Replace existing boxes with new box
            boxes_xyxy[i] = new_box
            
            # Remove overlapping boxes
            idxs = np.delete(idxs, np.concatenate(([0], overlapping_indices)))
        else:
            # If no overlapping boxes, remove only current box
            idxs = np.delete(idxs, 0)
    
    return boxes_xyxy[pick].tolist()

def process_svs_file(svs_path, output_dir):
    """
    Function to process SVS files and detect regions of interest
    
    Args:
        svs_path (str): SVS file path
        output_dir (str): Output directory for results
        
    Returns:
        tuple: (processed image, list of bounding box coordinates)
    """
    # Open SVS file
    slide = openslide.OpenSlide(svs_path)
    
    # Get image from the lowest resolution level
    level = slide.level_count - 1
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Convert black and dark gray areas to white
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)  # Lowered threshold to 60
    img[dark_mask == 0] = [255, 255, 255]
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Separate Saturation channel
    _, saturation, _ = cv2.split(hsv)
    
    # Normalize Saturation channel (to 0-255 range)
    saturation = cv2.normalize(saturation, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply Gaussian blur to remove noise
    saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    
    # Apply binarization to Saturation channel
    _, binary = cv2.threshold(saturation, 25, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to remove noise and connect regions
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Copy image for result visualization
    result_img = img.copy()
    debug_img = img.copy()  # Debug image
    
    # Store bounding box information
    bboxes = []
    min_area = 20000
    aspect_ratio_threshold = 0.1
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = min(w, h) / max(w, h)
        
        if area > min_area and aspect_ratio > aspect_ratio_threshold:
            bboxes.append([x, y, x+w, y+h])
    
    # Merge overlapping boxes
    merged_boxes = merge_overlapping_boxes(bboxes)
    
    # Sort boxes by x coordinate
    merged_boxes.sort(key=lambda box: box[0])  # Sort by x1 coordinate
    
    # Draw boxes on result image
    for i, box in enumerate(merged_boxes, 1):  # Assign numbers starting from 1
        x1, y1, x2, y2 = box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display box number (with background)
        text = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        # Draw text background
        cv2.rectangle(result_img, 
                     (x1, y1 - text_size[1] - 10),
                     (x1 + text_size[0], y1),
                     (0, 255, 0), -1)  # -1 for filled rectangle
        
        # Draw text
        cv2.putText(result_img, text,
                    (x1, y1 - 5),
                    font, font_scale, (0, 0, 0),  # Black text
                    font_thickness)
        
        # Draw same on debug image
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display number on debug image
        cv2.rectangle(debug_img,
                     (x1, y1 - text_size[1] - 10),
                     (x1 + text_size[0], y1),
                     (0, 255, 0), -1)
        cv2.putText(debug_img, text,
                    (x1, y1 - 5),
                    font, font_scale, (0, 0, 0),
                    font_thickness)
    
    # Save debug images
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    file_id = Path(svs_path).stem
    
    # Save original image
    cv2.imwrite(os.path.join(debug_dir, f"{file_id}_original.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # Save Saturation channel
    cv2.imwrite(os.path.join(debug_dir, f"{file_id}_saturation.png"), saturation)
    # Save binarization result
    cv2.imwrite(os.path.join(debug_dir, f"{file_id}_binary.png"), binary)
    # Save dark area mask
    cv2.imwrite(os.path.join(debug_dir, f"{file_id}_dark_mask.png"), dark_mask)
    # Save debug image (showing both original and merged boxes)
    cv2.imwrite(os.path.join(debug_dir, f"{file_id}_debug_boxes.png"), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    # Save Saturation histogram
    plt.figure(figsize=(10, 5))
    plt.hist(saturation.ravel(), bins=256, range=[0, 256])
    plt.title('Saturation Histogram')
    plt.savefig(os.path.join(debug_dir, f"{file_id}_histogram.png"))
    plt.close()
    
    return result_img, merged_boxes

def main():
    """
    Main execution function
    
    - Find all SVS files in input directory and process them
    - Perform region of interest detection for each file
    - Save results as images and Excel files
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect tissue regions in SVS files')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing SVS files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--recursive', action='store_true', default=True,
                        help='Search for SVS files recursively in subdirectories (default: True)')
    
    args = parser.parse_args()
    
    # Set input/output paths from arguments
    input_dir = args.input_dir
    output_base_dir = args.output_dir
    
    # Validate input directory
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Delete and recreate output directory
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all SVS files
    if args.recursive:
        svs_files = glob.glob(os.path.join(input_dir, "**/*.svs"), recursive=True)
    else:
        svs_files = glob.glob(os.path.join(input_dir, "*.svs"))
    
    if not svs_files:
        print(f"No SVS files found in {input_dir}")
        return
    
    print(f"Found {len(svs_files)} SVS files to process")
    
    # Create DataFrame to store results
    results = []
    infos = []
    
    # Process each SVS file (show progress with tqdm)
    for svs_path in tqdm(svs_files, desc="Processing SVS files"):
        # Extract file ID (without extension)
        file_id = Path(svs_path).stem
        
        # Get SVS file information
        slide = openslide.OpenSlide(svs_path)
        
        # Get mpp information (use only x direction)
        mpp = float(slide.properties.get('openslide.mpp-x', '0'))
        
        # Relationship between mpp and magnification:
        # magnification = 10 / mpp
        # Examples:
        # - mpp = 0.5 → 10 / 0.5 = 20x
        # - mpp = 0.25 → 10 / 0.25 = 40x
        # - mpp = 1.0 → 10 / 1.0 = 10x
        # Note: This is a general standard and may vary slightly depending on scanner or settings
        
        # Collect resolution information by magnification
        mag_info = {}
        mag_levels = []  # List to store level information
        
        # Check level_downsamples
        downsamples = slide.level_downsamples
        print(f"\nFile: {file_id}")
        print(f"MPP: {mpp:.4f}")
        print(f"Level downsamples: {downsamples}")
        
        for level in range(slide.level_count):
            width, height = slide.level_dimensions[level]
            downsample = downsamples[level]
            # Calculate mpp for each level
            current_mpp = mpp * downsample
            mag_levels.append((level, width, height, current_mpp))
            print(f"Level {level}: {width}x{height} (mpp: {current_mpp:.4f})")
        
        # Save information in level order
        for level, width, height, current_mpp in mag_levels:
            mag_info[f'width_level_{level}'] = width
            mag_info[f'height_level_{level}'] = height
            mag_info[f'mpp_level_{level}'] = current_mpp
        
        # Find annotation file path
        anno_path = None
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.startswith(file_id) and file.endswith('.qpdata'):
                    anno_path = os.path.join(root, file)
                    break
            if anno_path:
                break
        
        # Convert to relative paths
        rel_svs_path = os.path.relpath(svs_path, input_dir)
        rel_anno_path = os.path.relpath(anno_path, input_dir) if anno_path else 'Not found'
        
        # Extract folder name
        folder_name = os.path.dirname(rel_svs_path)
        
        # Add information to infos DataFrame
        info_dict = {
            'ID': file_id,
            'label': folder_name,
            'filepath_img': rel_svs_path,
            'filepath_anno': rel_anno_path,
            'mpp': mpp,  # Base mpp value
            'total_levels': slide.level_count
        }
        # Add level-specific information
        info_dict.update(mag_info)
        infos.append(info_dict)
        
        # Get result image and bounding box information
        result_img, bboxes = process_svs_file(svs_path, output_base_dir)
        
        # Save result image
        output_img_path = os.path.join(output_base_dir, 'images', f"{file_id}.png")
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        
        # Save bounding box information
        for i, bbox in enumerate(bboxes, 1):  # Assign numbers starting from 1
            results.append({
                'ID': file_id,
                'tissue_no': i,  # Add box sequence number
                'xmin': bbox[0],
                'xmax': bbox[2],
                'ymin': bbox[1],
                'ymax': bbox[3]
            })
    
    # Convert results to DataFrame and save as Excel file
    df = pd.DataFrame(results).sort_values(by=['ID', 'tissue_no']).reset_index(drop=True)
    output_excel_path = os.path.join(output_base_dir, "bboxes.xlsx")
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    
    # Save infos DataFrame as Excel file
    infos_df = pd.DataFrame(infos)
    infos_excel_path = os.path.join(output_base_dir, "infos.xlsx")
    infos_df.to_excel(infos_excel_path, index=False, engine='openpyxl')
    
    print(f"Processing completed. Results saved to {output_base_dir}")
    print(f"Total tissue regions detected: {len(results)}")

if __name__ == "__main__":
    main()
