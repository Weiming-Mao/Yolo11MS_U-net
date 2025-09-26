# -*- coding: utf-8 -*-
"""
SEM Grid Defect Detection (Integrated Folder Batch Evaluation - Final Fixed Version)
Dependencies: opencv-python, numpy, pandas, tqdm
pip install opencv-python numpy pandas tqdm
"""

import cv2
import numpy as np
import json
import os
import glob
from tqdm import tqdm
import pandas as pd

def create_masks_from_json(json_path, class_map={'stain': 'contamination', 'damage': 'damage'}):
    """
    Creates multi-class ground truth masks from an ISAT-format JSON annotation file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file not found: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    width = data['info']['width']
    height = data['info']['height']
    
    gt_masks = {
        'contamination': np.zeros((height, width), dtype=np.uint8),
        'damage': np.zeros((height, width), dtype=np.uint8)
    }
    
    for obj in data['objects']:
        category = obj.get('category')
        internal_category = class_map.get(category)
        
        if internal_category in gt_masks:
            segmentation = np.array(obj['segmentation'], dtype=np.int32)
            cv2.fillPoly(gt_masks[internal_category], [segmentation], 255)
            
    return gt_masks, (height, width)

def calculate_segmentation_metrics(pred_mask, gt_mask):
    """
    Calculates segmentation metrics (IoU, Precision, Recall) for a single class.
    """
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    
    intersection = np.logical_and(pred_bool, gt_bool)
    union = np.logical_or(pred_bool, gt_bool)
    
    tp = np.sum(intersection)
    fp = np.sum(np.logical_and(pred_bool, ~gt_bool))
    fn = np.sum(np.logical_and(~pred_bool, gt_bool))

    iou = tp / (np.sum(union) + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return {'iou': iou, 'precision': precision, 'recall': recall}


def preprocess(img_gray, gauss_ksize=5, clahe_clip=2.0, clahe_tile=8):
    """Preprocessing: Gaussian blur and optional CLAHE enhancement"""
    img_blur = cv2.GaussianBlur(img_gray, (gauss_ksize, gauss_ksize), 0)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    img_clahe = clahe.apply(img_blur)
    return img_blur, img_clahe

def remove_small_components(bin_mask, min_area=50):
    """Removes small noise components via connected components analysis. Returns clean_mask"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    clean_mask = np.zeros_like(bin_mask, dtype=np.uint8)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == lab] = 255
    return clean_mask


def detect_contamination_tophat(img_blur, tophat_ksize=25, threshold=30, min_area=20, morph_ksize=3):
    """[Improved Method] Detects bright spot contamination using Top-hat transform (local anomaly detection)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize))
    tophat_img = cv2.morphologyEx(img_blur, cv2.MORPH_TOPHAT, kernel)
    _, thr = cv2.threshold(tophat_img, threshold, 255, cv2.THRESH_BINARY)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    clean_mask = remove_small_components(thr, min_area=min_area)
    return clean_mask, tophat_img

# ================================== FIX STARTS HERE ==================================
def detect_damage(img_clahe, percentile=1.5, morph_kernel=9, min_area=200):
    """
    [FIXED] Detects dark area damage.
    Changed the parameter 'dark_percentile' to 'percentile' to match the calling code.
    """
    thresh_val = np.percentile(img_clahe, percentile)
    _, thr = cv2.threshold(img_clahe, float(thresh_val), 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    clean_mask = remove_small_components(thr, min_area=min_area)
    return clean_mask
# =================================== FIX ENDS HERE ===================================

def overlay_results(img_gray, contamination_mask, damage_mask, alpha=0.6, output_path=None):
    """Overlays results onto the original image and optionally saves it. Colors: Contamination=Red, Damage=Blue"""
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    overlay[contamination_mask > 0] = (0, 0, 255) # Red
    overlay[damage_mask > 0] = (255, 0, 0)       # Blue
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    
    if output_path:
        cv2.imwrite(output_path, out)
        
    return out

def analyze_and_evaluate_single_image(img_path, json_path, params):
    """
    Detects and evaluates a single image, returning a dictionary of metrics.
    """
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Warning: Could not read image {img_path}")
        return None

    # Detection pipeline
    img_blur, img_clahe = preprocess(img_gray, **{k:v for k,v in params.items() if k.startswith('gauss') or k.startswith('clahe')})
    pred_cont_mask, _ = detect_contamination_tophat(img_blur, **{k.replace('cont_', ''):v for k,v in params.items() if k.startswith('cont')})
    pred_dmg_mask = detect_damage(img_clahe, **{k.replace('dark_', ''):v for k,v in params.items() if k.startswith('dark')})

    # Evaluation pipeline
    try:
        gt_masks, (h, w) = create_masks_from_json(json_path)
    except FileNotFoundError:
        print(f"Warning: Corresponding annotation file not found {json_path}")
        return None

    # Ensure prediction mask and ground truth mask have the same dimensions
    if pred_cont_mask.shape != (h, w):
        pred_cont_mask = cv2.resize(pred_cont_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        pred_dmg_mask = cv2.resize(pred_dmg_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    cont_metrics = calculate_segmentation_metrics(pred_cont_mask, gt_masks['contamination'])
    dmg_metrics = calculate_segmentation_metrics(pred_dmg_mask, gt_masks['damage'])
    
    miou = (cont_metrics['iou'] + dmg_metrics['iou']) / 2
    
    return {
        'filename': os.path.basename(img_path),
        'cont_iou': cont_metrics['iou'],
        'cont_precision': cont_metrics['precision'],
        'cont_recall': cont_metrics['recall'],
        'dmg_iou': dmg_metrics['iou'],
        'dmg_precision': dmg_metrics['precision'],
        'dmg_recall': dmg_metrics['recall'],
        'mIoU': miou
    }


# --- Main Program: Batch Evaluate Test Set Folder ---
if __name__ == "__main__":
    # --- 1. Configure Parameters and Paths ---
    
    # !!! Please change this to your test set image folder path !!!
    TEST_IMAGE_DIR = "test0915/images"
    
    # !!! Please change this to your test set annotation folder path !!!
    TEST_ANNOTATION_DIR = "test0915/labels"
    
    # (Optional) Set a folder to save visualization images with overlaid detection results
    OUTPUT_VISUALIZATION_DIR = "out/test_results"
    
    # Ensure the output directory exists
    if OUTPUT_VISUALIZATION_DIR:
        os.makedirs(OUTPUT_VISUALIZATION_DIR, exist_ok=True)

    # Algorithm hyperparameters
    PARAMS = {
        'gauss_ksize': 5,
        'clahe_clip': 2.0,
        'clahe_tile': 8,
        'cont_tophat_ksize': 31,
        'cont_threshold': 40,
        'cont_min_area': 25,
        'dark_percentile': 1.5,
        'dark_min_area': 300,
        'dark_morph_kernel': 9
    }
    
    # --- 2. Find All Test Images and Start Evaluation ---
    
    # Use glob to find all jpg files
    image_paths = sorted(glob.glob(os.path.join(TEST_IMAGE_DIR, '*.jpg')))
    if not image_paths:
        print(f"Error: No .jpg images found in the folder '{TEST_IMAGE_DIR}'.")
        exit()

    all_metrics = []
    print(f"Starting evaluation of {len(image_paths)} images in the test set...")

    # Use tqdm to display a progress bar
    for img_path in tqdm(image_paths, desc="Evaluation Progress"):
        basename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(basename)[0]
        json_path = os.path.join(TEST_ANNOTATION_DIR, f"{filename_no_ext}.json")
        
        # Detect and evaluate a single image
        metrics = analyze_and_evaluate_single_image(img_path, json_path, PARAMS)
        
        if metrics:
            all_metrics.append(metrics)
            
            # (Optional) Generate and save visualization results
            if OUTPUT_VISUALIZATION_DIR:
                img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_blur, img_clahe = preprocess(img_gray, **{k:v for k,v in PARAMS.items() if k.startswith('gauss') or k.startswith('clahe')})
                pred_cont_mask, _ = detect_contamination_tophat(img_blur, **{k.replace('cont_', ''):v for k,v in PARAMS.items() if k.startswith('cont')})
                pred_dmg_mask = detect_damage(img_clahe, **{k.replace('dark_', ''):v for k,v in PARAMS.items() if k.startswith('dark')})
                
                output_path = os.path.join(OUTPUT_VISUALIZATION_DIR, basename)
                overlay_results(img_gray, pred_cont_mask, pred_dmg_mask, output_path=output_path)


    # --- 3. Calculate and Print Final Average Evaluation Metrics ---

    if not all_metrics:
        print("Evaluation produced no results. Please check file paths and if files match.")
    else:
        # Use a pandas DataFrame for convenient calculations
        df_metrics = pd.DataFrame(all_metrics)
        
        # Calculate the average
        avg_metrics = df_metrics.mean(numeric_only=True)
        
        print("\n" + "="*50)
        print(f"Batch evaluation of test set complete ({len(df_metrics)} images)")
        print("="*50)
        print(f"Algorithm Parameters: {PARAMS}")
        print("-"*50)
        print(f"Contamination Average Metrics:")
        print(f"  IoU:        {avg_metrics['cont_iou']:.4f}")
        print(f"  Precision:  {avg_metrics['cont_precision']:.4f}")
        print(f"  Recall:     {avg_metrics['cont_recall']:.4f}")
        
        print(f"\nDamage Average Metrics:")
        print(f"  IoU:        {avg_metrics['dmg_iou']:.4f}")
        print(f"  Precision:  {avg_metrics['dmg_precision']:.4f}")
        print(f"  Recall:     {avg_metrics['dmg_recall']:.4f}")
        
        print("\n--- Overall Performance ---")
        print(f"Average mIoU (Mean IoU): {avg_metrics['mIoU']:.4f}")
        print("="*50)

        # (Optional) Save detailed results to a CSV file
        csv_output_path = "out/test_set_detailed_metrics.csv"
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        df_metrics.to_csv(csv_output_path, index=False, float_format='%.4f')
        print(f"Detailed evaluation results saved to: {csv_output_path}")