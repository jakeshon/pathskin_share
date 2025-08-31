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
import argparse

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Classify patches as background or foreground')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing patch data (output from Stage 2)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--file_id', type=str, default=None,
                        help='Specific file_id to process (if not specified, processes all)')
    parser.add_argument('--std_threshold', type=float, default=40.0,
                        help='Standard deviation threshold for background classification (default: 40.0)')
    parser.add_argument('--edge_threshold', type=float, default=15.0,
                        help='Edge intensity threshold for background classification (default: 15.0)')
    parser.add_argument('--output_suffix', type=str, default='test',
                        help='Suffix for output files (default: test)')
    
    args = parser.parse_args()
    
    # Set base folder and output directory
    base_dir = args.base_dir
    output_dir = args.output_dir
    
    # Validate base directory
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read Excel file
    excel_path = os.path.join(base_dir, "bboxes_patch_20x.xlsx")
    if not os.path.exists(excel_path):
        raise ValueError(f"Excel file not found: {excel_path}")
    
    df = pd.read_excel(excel_path, engine='openpyxl')
    
    if df.empty:
        print("No patch data found in the Excel file")
        return
    
    # Filter by file_id if specified
    if args.file_id:
        test_df = df[df['file_id'] == args.file_id]
        if test_df.empty:
            print(f"No patches found for file_id: {args.file_id}")
            return
    else:
        test_df = df
    
    print(f"Processing {len(test_df)} patches from {test_df['file_id'].nunique()} files")
    
    # Set thresholds
    thresholds = {
        'std': args.std_threshold,
        'edge': args.edge_threshold
    }
    
    print(f"Using thresholds - std: {thresholds['std']}, edge: {thresholds['edge']}")
    
    # Process each file_id separately
    all_results = []
    
    for file_id in test_df['file_id'].unique():
        file_df = test_df[test_df['file_id'] == file_id]
        
        # Find mask image for this file_id
        mask_path = None
        for tissue_no in file_df['tissue_no'].unique():
            potential_mask_path = os.path.join(base_dir, "tissues", file_id, f"tissue{tissue_no}_mask.jpg")
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break
        
        if mask_path is None:
            print(f"Warning: No mask image found for {file_id}")
            continue
        
        # Read mask image
        mask_image = cv2.imread(mask_path)
        if mask_image is None:
            print(f"Cannot read mask image: {mask_path}")
            continue
        
        print(f"Processing {file_id} with {len(file_df)} patches")
        
        # Results for this file
        file_results = []
        error_count = 0
        
        # Process each patch for this file
        for idx, row in tqdm(file_df.iterrows(), total=len(file_df), desc=f"Processing {file_id}"):
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
                file_results.append({
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
        
        # Save mask image for this file
        output_mask_path = os.path.join(output_dir, f"{file_id}_mask_with_fg.jpg")
        cv2.imwrite(output_mask_path, mask_image)
        
        # Add file results to all results
        all_results.extend(file_results)
        
        print(f"Completed {file_id}: {len(file_results)} patches processed, {error_count} errors")
    
    # Convert all results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("No patches were processed successfully")
        return
    
    # Output results
    print("\nProcessing results:")
    print(results_df)
    
    # Output background/foreground ratio
    bg_count = results_df['is_background'].sum()
    fg_count = len(results_df) - bg_count
    print(f"\nBackground images: {bg_count} ({bg_count/len(results_df)*100:.1f}%)")
    print(f"Foreground images: {fg_count} ({fg_count/len(results_df)*100:.1f}%)")
    
    # Save results as Excel file
    output_path = os.path.join(output_dir, f"bboxes_bg_fg_{args.output_suffix}.xlsx")
    results_df.to_excel(output_path, index=False)
    print(f"\nResults have been saved to {output_path}")
    print(f"Mask images have been saved to {output_dir}")

if __name__ == "__main__":
    main()
