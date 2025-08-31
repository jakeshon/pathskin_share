"""
This script is a program for extracting tissue sections at 20x magnification resolution from SVS files and dividing them into patches.

Main features:
1. Read tissue location information from Excel files
2. Extract 20x magnification resolution images from SVS files
3. Generate masks based on GeoJSON files
4. Divide tissue sections into 512x512 sized patches
5. Save each patch and its corresponding mask
6. Generate visualization images with patch numbers displayed (original and mask overlay)
7. Save patch information as Excel files

Output files:
- patches/: Each patch image (JPG format)
- masks/: Mask image for each patch (PNG format)
- tissues/: Visualization images with patch numbers displayed (original and mask overlay)
- bboxes_patch_20x.xlsx: Patch location and information

Usage:
1. SVS files must be in the input directory.
2. bboxes.xlsx file must contain tissue location information.
3. Each SVS file must have a corresponding GeoJSON file.
4. Run the script to automatically extract and save patches and masks.
"""

import shutil
import openslide
import numpy as np
import cv2
import pandas as pd
import os
import json
from pathlib import Path
from tqdm import tqdm

def get_level_for_magnification(slide, target_magnification):
    """
    Function to find the level corresponding to the target magnification
    
    Args:
        slide: OpenSlide object
        target_magnification: Target magnification (e.g., 20)
        
    Returns:
        int: Corresponding level number
    """
    # Check base magnification of slide
    base_magnification = float(slide.properties.get('openslide.objective-power', 0))
    if base_magnification == 0:
        raise ValueError("Cannot determine base magnification of slide.")
    
    # Calculate magnification for each level
    for level in range(slide.level_count):
        level_magnification = base_magnification / (2 ** level)
        if abs(level_magnification - target_magnification) < 1:
            return level
    
    raise ValueError(f"Cannot find level corresponding to {target_magnification}x magnification.")

def create_mask_from_geojson(geojson_path, svs_path):
    # Open SVS file
    slide = openslide.OpenSlide(svs_path)
    width, height = slide.dimensions
    
    # Read GeoJSON file
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    # Handle case where data is an array of features
    if isinstance(data, list):
        features = data
    else:
        features = data['features']
    
    # Set image size (same as SVS size)
    image_size = (height, width)
    
    # Create empty mask
    mask = np.zeros(image_size, dtype=np.uint8)
    
    # Draw mask for each feature
    for feature in features:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            coords = np.array(coords)
            coords = coords.astype(np.int32)
            
            # Draw polygon on mask
            cv2.fillPoly(mask, [coords], 255)
        elif feature['geometry']['type'] == 'MultiPolygon':
            for polygon in feature['geometry']['coordinates']:
                coords = polygon[0]
                coords = np.array(coords)
                coords = coords.astype(np.int32)
                
                # Draw polygon on mask
                cv2.fillPoly(mask, [coords], 255)
    
    return mask

def extract_tissue_patch(svs_path, bbox, output_dir, file_id, tissue_no, target_magnification=20):
    # Open SVS file
    slide = openslide.OpenSlide(svs_path)
    
    # Find level corresponding to target magnification
    target_level = get_level_for_magnification(slide, target_magnification)
    
    # Lowest resolution level
    lowest_level = slide.level_count - 1
    
    # Bounding box coordinates (based on lowest resolution)
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate size ratio between level 0 and lowest resolution
    level0_width, level0_height = slide.level_dimensions[0]
    lowest_width, lowest_height = slide.level_dimensions[lowest_level]
    
    width_ratio = level0_width / lowest_width
    height_ratio = level0_height / lowest_height
    
    # Calculate level 0 coordinates
    xmin0 = int(xmin * width_ratio)
    ymin0 = int(ymin * height_ratio)
    xmax0 = int(xmax * width_ratio)
    ymax0 = int(ymax * height_ratio)
    
    # Set patch size and step size
    patch_size = 512
    step_size = patch_size // 2  # 256
    
    # List to store patch information
    patch_info_list = []
    
    # Find GeoJSON file path
    geojson_path = None
    for root, _, files in os.walk(os.path.dirname(svs_path)):
        for file in files:
            if file.startswith(file_id) and file.endswith('.geojson'):
                geojson_path = os.path.join(root, file)
                break
        if geojson_path:
            break
    
    if geojson_path is None:
        print(f"Warning: Cannot find GeoJSON file corresponding to {file_id}.")
        return []
    
    # Create full mask
    tissues_dir = os.path.join(output_dir, 'tissues', file_id)
    os.makedirs(tissues_dir, exist_ok=True)
    mask_path = os.path.join(tissues_dir, f"tissue{tissue_no}_mask.jpg")
    full_mask = create_mask_from_geojson(geojson_path, svs_path)
    
    # Get 20x magnification image (for visualization)
    vis_level = target_level
    vis_img = slide.read_region((0, 0), vis_level, slide.level_dimensions[vis_level])
    vis_img = np.array(vis_img)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGBA2RGB)
    
    # Mark mask area in blue
    overlay = vis_img.copy()
    overlay[full_mask > 0] = [0, 0, 255]  # Blue (BGR)
    
    # Composite mask semi-transparently
    alpha = 0.3  # Transparency
    vis_mask = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)
    
    # Initialize patch number
    patch_no = 1
    
    # Calculate patch positions and visualize
    for y in range(ymin0 - step_size, ymax0 + step_size, step_size):
        for x in range(xmin0 - step_size, xmax0 + step_size, step_size):
            # Adjust so patch doesn't exceed image boundaries
            if x < 0 or y < 0 or x + patch_size > level0_width or y + patch_size > level0_height:
                continue
            
            # Mark patch area (image and mask)
            cv2.rectangle(vis_img, 
                         (x, y), 
                         (x + patch_size, y + patch_size), 
                         (0, 255, 0), 2)
            
            cv2.rectangle(vis_mask, 
                         (x, y), 
                         (x + patch_size, y + patch_size), 
                         (0, 255, 0), 2)
            
            # Display patch number
            text = str(patch_no)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            
            # Calculate text position (left top of box)
            text_x = x + 10  # 10 pixel margin from left
            text_y = y + text_size[1] + 10  # 10 pixel margin from top
            
            # Draw text background (image)
            padding = 5
            bg_x1 = text_x - padding
            bg_y1 = text_y - text_size[1] - padding
            bg_x2 = text_x + text_size[0] + padding
            bg_y2 = text_y + padding
            
            # Draw background (image)
            cv2.rectangle(vis_img, 
                         (bg_x1, bg_y1),
                         (bg_x2, bg_y2),
                         (255, 255, 255), -1)  # White background
            
            # Draw text (image)
            cv2.putText(vis_img, text,
                       (text_x, text_y),
                       font, font_scale, (0, 0, 0),  # Black text
                       font_thickness)
            
            # Draw background (mask)
            cv2.rectangle(vis_mask, 
                         (bg_x1, bg_y1),
                         (bg_x2, bg_y2),
                         (255, 255, 255), -1)  # White background
            
            # Draw text (mask)
            cv2.putText(vis_mask, text,
                       (text_x, text_y),
                       font, font_scale, (0, 0, 0),  # Black text
                       font_thickness)
            
            patch_no += 1
    
    # Save 20x magnification image
    cv2.imwrite(os.path.join(tissues_dir, f"tissue{tissue_no}_patches.jpg"), 
                cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Save image with mask overlay
    cv2.imwrite(mask_path, 
                cv2.cvtColor(vis_mask, cv2.COLOR_RGB2BGR), 
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Extract and save patches
    patch_no = 1
    for y in range(ymin0 - step_size, ymax0 + step_size, step_size):
        for x in range(xmin0 - step_size, xmax0 + step_size, step_size):
            # Adjust so patch doesn't exceed image boundaries
            if x < 0 or y < 0 or x + patch_size > level0_width or y + patch_size > level0_height:
                continue
                
            # Extract patch
            # img = slide.read_region((x, y), target_level, (patch_size, patch_size))
            # img = np.array(img)
            # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Generate patch filename
            patch_filename = f"{file_id}_tissue{tissue_no}_patch{patch_no}.jpg"
            save_dir = os.path.join(output_dir, 'patches', file_id, str(tissue_no))
            os.makedirs(save_dir, exist_ok=True)
            patch_path = os.path.join(save_dir, patch_filename)
            
            # Save image (in JPG format)
            cv2.imwrite(patch_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Extract and save patch mask
            patch_mask = full_mask[y:y+patch_size, x:x+patch_size]
            
            
            # Create relative path (excluding output_dir)
            relative_path = os.path.join('patches', file_id, str(tissue_no), patch_filename)
            
            # Save only if mask area exists
            if np.any(patch_mask > 0):
                mask_filename = f"{file_id}_tissue{tissue_no}_patch{patch_no}_mask.png"
                mask_save_dir = os.path.join(output_dir, 'masks', file_id, str(tissue_no))
                os.makedirs(mask_save_dir, exist_ok=True)
                mask_path = os.path.join(mask_save_dir, mask_filename)
                cv2.imwrite(mask_path, patch_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                
                # Create relative path (excluding output_dir)
                relative_mask_path = os.path.join('patches', file_id, str(tissue_no), mask_filename)
            else:
                relative_mask_path = None
            
            # Save patch information
            patch_info = {
                'file_id': file_id,
                'tissue_no': tissue_no,
                'patch_no': patch_no,
                'patch_filename': patch_filename,
                'filepath_img': relative_path,  # Save only relative path
                'filepath_anno': relative_mask_path,  # Save mask relative path
                'x': x,
                'y': y,
                'xmin0': xmin0,
                'ymin0': ymin0,
                'xmax0': xmax0,
                'ymax0': ymax0
            }
            patch_info_list.append(patch_info)
            
            patch_no += 1
    
    return patch_info_list

def main():
    # Set input/output paths
    input_dir = "/Users/shon/ws/ws_proj/research/pathskin/data/2025년 5월 CM pathology annotated data"
    bbox_excel = "/Users/shon/ws/ws_proj/research/pathskin/output/ex01_02/1.find_tissue/bboxes.xlsx"
    output_dir = "/Users/shon/ws/ws_proj/research/pathskin/output/ex01_02/2.crop_patch_20x"
    
    # Create output directory
    # shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(output_dir + "/masks", ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read Excel file
    df = pd.read_excel(bbox_excel)
    
    # List to store patch information
    patch_info_list = []
    
    # Process each tissue section
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting tissue sections"):
        file_id = row['ID']
        tissue_no = row['tissue_no']
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        
        # Find SVS file path
        svs_path = None
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.startswith(file_id) and file.endswith('.svs'):
                    svs_path = os.path.join(root, file)
                    break
            if svs_path:
                break
        
        if svs_path is None:
            print(f"Warning: Cannot find SVS file corresponding to {file_id}.")
            continue
        
        try:
            # Extract patches
            patch_info = extract_tissue_patch(svs_path, bbox, output_dir, file_id, tissue_no)
            patch_info_list.extend(patch_info)
        except Exception as e:
            print(f"Error processing {file_id} tissue {tissue_no}: {str(e)}")
    
    # Convert patch information to DataFrame
    patch_df = pd.DataFrame(patch_info_list)
    patch_df.to_excel(os.path.join(output_dir, 'bboxes_patch_20x.xlsx'), index=False)
    
    print("\nPatch information has been saved to bboxes_patch_20x.xlsx file.")
    print("\nPatch information summary:")
    print(f"Total patches: {len(patch_df)}")
    print(patch_df.head())

if __name__ == "__main__":
    main()
