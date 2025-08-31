"""
Background/Foreground separation program for patch images

Features:
- Read patch image information for specific file_id from Excel file
- Calculate statistics (mean, standard deviation, edge intensity) for each patch image
- Separate background/foreground based on statistics
- Display 'FG' text on foreground patch locations
- Save results as Excel file

Input:
- Excel file (bboxes_patch_20x.xlsx)
- Mask image (tissue1_mask.jpg)

Output:
- Mask image with 'FG' text displayed
- Background/foreground separation result Excel file
"""

import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def calculate_image_statistics(image):
    """Calculate image statistics
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        dict: Mean, standard deviation, edge intensity
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate basic statistics
    mean = np.mean(gray)  # Mean brightness value
    std = np.std(gray)    # Standard deviation (degree of pixel value variance)
    
    # Edge detection (using Canny edge detection algorithm)
    edges = cv2.Canny(gray, 100, 200)  # Edge detection with thresholds 100, 200
    edge_intensity = np.mean(edges)    # Mean value of edge intensity
    
    return {
        'mean': mean,
        'std': std,
        'edge_intensity': edge_intensity
    }

def is_background(stats, thresholds):
    """Background determination based on statistics
    
    Args:
        stats: Image statistics
        thresholds: Threshold settings
    
    Returns:
        bool: True if background, False if foreground
    """
    # Background determination criteria:
    # 1. Low standard deviation (uniform pixel values)
    # 2. Low edge intensity
    return (stats['std'] < thresholds['std']) and (stats['edge_intensity'] < thresholds['edge'])

def put_text_with_outline(img, text, position, font_scale, thickness):
    """Draw text with outline
    
    Args:
        img: Image
        text: Text to display
        position: Text position (x, y)
        font_scale: Font size
        thickness: Line thickness
    """
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Adjust text position (slight offset from top-left of patch)
    x, y = position
    x = x + 15  # Left margin
    y = y + text_height + 50  # Top margin
    
    # Draw black outline
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        cv2.putText(img, text, (x+dx, y+dy), font, font_scale, (0,0,0), thickness+2)
    
    # Draw green text
    cv2.putText(img, text, (x, y), font, font_scale, (0,255,0), thickness)

def main():
    """Main function
    
    Processing steps:
    1. Read patch information from Excel file
    2. Load mask image
    3. Process each patch image (calculate statistics, separate background/foreground)
    4. Save results (mask image, Excel file)
    """
    # Set base folder
    base_dir = "/Users/shon/ws/ws_proj/research/pathskin/output/ex01_02/2.crop_patch_20x"
    output_dir = "/Users/shon/ws/ws_proj/research/pathskin/output/ex01_02/3.find_bg_fg"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read Excel file
    excel_path = os.path.join(base_dir, "bboxes_patch_20x.xlsx")
    df = pd.read_excel(excel_path, engine='openpyxl')
    
    # Select only images with file_id C3L-00967-21
    test_df = df[df['file_id'] == 'C3L-00967-21']
    print(f"Processing {len(test_df)} images in total.")
    
    # Read mask image
    mask_path = os.path.join(base_dir, "tissues/C3L-00967-21/tissue1_mask.jpg")
    mask_image = cv2.imread(mask_path)
    if mask_image is None:
        print(f"Cannot read mask image: {mask_path}")
        return
    
    # Set thresholds
    thresholds = {
        'std': 40,  # Standard deviation threshold
        'edge': 15  # Edge intensity threshold
    }
    
    # List to store results
    results = []
    error_count = 0
    
    # Process each image
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing images"):
        try:
            # Generate image path
            image_path = os.path.join(base_dir, row['filepath_img'])
            
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Cannot read image: {image_path}")
                error_count += 1
                continue
                
            # Calculate image statistics
            stats = calculate_image_statistics(image)
            
            # Determine if background
            is_bg = is_background(stats, thresholds)
            
            # Save results
            results.append({
                'file_id': row['file_id'],
                'tissue_no': row['tissue_no'],
                'patch_no': row['patch_no'],
                'image_path': image_path,
                'mean': stats['mean'],
                'std': stats['std'],
                'edge_intensity': stats['edge_intensity'],
                'is_background': is_bg
            })
            
            # Display text on mask image if foreground
            if not is_bg:
                x, y = row['x'], row['y']
                put_text_with_outline(mask_image, "FG", (int(x), int(y)), 
                                    font_scale=1.0, thickness=2)
            
        except Exception as e:
            print(f"Error occurred while processing image: {image_path}")
            print(f"Error details: {str(e)}")
            error_count += 1
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No images were processed.")
        return
    
    # Output results
    print("\nProcessing results:")
    print(results_df)
    
    # Output background/foreground ratio
    bg_count = results_df['is_background'].sum()
    fg_count = len(results_df) - bg_count
    print(f"\nBackground images: {bg_count} ({bg_count/len(results_df)*100:.1f}%)")
    print(f"Foreground images: {fg_count} ({fg_count/len(results_df)*100:.1f}%)")
    print(f"Processing failures: {error_count}")
    
    # Save mask image
    output_mask_path = os.path.join(output_dir, "tissue1_mask_with_fg.jpg")
    cv2.imwrite(output_mask_path, mask_image)
    print(f"\nMask image has been saved to {output_mask_path}.")
    
    # Save results as Excel file
    output_path = os.path.join(output_dir, "bboxes_bg_fg_test.xlsx")
    results_df.to_excel(output_path, index=False)
    print(f"Results have been saved to {output_path}.")

if __name__ == "__main__":
    main()
