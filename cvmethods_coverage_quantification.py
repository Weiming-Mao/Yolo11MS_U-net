# -*- coding: utf-8 -*-
"""
Fully Automated Graphene Grid Coverage Analysis Script (V6 - Dual Threshold Connected Components Method)

Core Idea:
- Strictly follows user instructions, using connected components as the core analysis tool.
- Removes extra constraints like circularity, filtering solely based on area.
- Employs a dual-threshold strategy:
  1. A low threshold + connected components to precisely find "uncovered" black holes.
  2. A high threshold + connected components to find "all" holes (black + gray).
  3. By comparing positions, it subtracts "uncovered holes" from "all holes" to get "covered" gray holes.
- Retains and upgrades the multi-stage visual diagnostic tool to adapt to the new algorithm.
"""
import cv2
import numpy as np
import os

# ============================================================================== 
# 1. User Parameter Configuration Area (V6)
# ============================================================================== 
class Config:
    # --- 1. Grid Region Extraction Parameters ---
    REGION_MIN_AREA = 40000; REGION_MAX_AREA = 750000
    REGION_MIN_ASPECT_RATIO = 0.7; REGION_MAX_ASPECT_RATIO = 1.3
    REGION_MIN_SOLIDITY = 0.90

    # --- 2. Hole Detection Parameters (Dual Threshold Connected Components) ---
    # [CRITICAL DEBUGGING POINT 1] Strict threshold for identifying "uncovered" black holes (0-255)
    # The lower the value, the darker a region must be to be identified.
    UNCOVERED_HOLE_THRESHOLD = 65

    # [CRITICAL DEBUGGING POINT 2] Lenient threshold for identifying "all" holes
    # This value should be higher than the grayscale of black holes but lower than that of the grid skeleton.
    # Set to "auto" to attempt automatic determination, or manually specify a value from 0-255.
    ALL_HOLES_THRESHOLD = "auto"
    # If set to "auto", this offset is used to calculate the threshold from the histogram peak.
    ALL_HOLES_THRESHOLD_OFFSET = 15 

    # Dynamic ratio for hole area filtering (no circularity).
    HOLE_MAX_AREA_RATIO_OF_REGION = 1 / 400.0
    HOLE_MIN_AREA_RATIO_OF_REGION = 1 / 10000.0

    # --- 3. Visualization Parameters ---
    # Note: OpenCV uses BGR format (Blue, Green, Red)
    COLOR_REGION = (255, 0, 0)      # Region contour color (currently blue)
    COLOR_COVERED = (255, 0, 0)     # Covered holes -> changed to blue (BGR)
    COLOR_UNCOVERED = (0, 0, 255)   # Uncovered holes -> remains red (BGR)


# ============================================================================== 
# 2. Core Function Definitions
# ============================================================================== 

def create_debug_visualization_v6(stages, final_contours, title):
    """[V6 Version - Fixed] Diagnostic image generator."""
    target_height = 300
    vis_images = []

    for name, img in stages.items():
        if img is None: continue
        h, w = img.shape[:2]
        scale = target_height / h
        resized_img = cv2.resize(img, (int(w * scale), target_height))
        # Ensure all base stage images are in 3D BGR format
        if len(resized_img.shape) == 2:
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(resized_img, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        vis_images.append(resized_img)

    # Prepare the final result image
    final_result_img = stages["1. Original"].copy()

    # --- BUG FIX START ---
    # Regardless of whether contours are found, first convert the final result image to BGR to ensure dimensional consistency.
    if len(final_result_img.shape) == 2:
        final_result_img = cv2.cvtColor(final_result_img, cv2.COLOR_GRAY2BGR)
    # --- BUG FIX END ---

    # Now it's safe to draw colored contours
    cv2.drawContours(final_result_img, final_contours['covered'], -1, Config.COLOR_COVERED, -1)
    cv2.drawContours(final_result_img, final_contours['uncovered'], -1, Config.COLOR_UNCOVERED, -1)

    # Resize and add a label
    final_result_img = cv2.resize(final_result_img, (vis_images[0].shape[1], vis_images[0].shape[0]))
    cv2.putText(final_result_img, "4. Final Result", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    vis_images.append(final_result_img)

    # Horizontally stack all images
    composite_image = np.hstack(vis_images)
    cv2.putText(composite_image, title, (10, composite_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return composite_image

def get_components_from_threshold(gray_image, mask, threshold, area_range):
    """Extracts connected components based on a threshold and area filtering."""
    # Inverse binary thresholding: pixels below the threshold (dark areas) become 255 (white)
    _, binary_img = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)
    binary_img = cv2.bitwise_and(binary_img, binary_img, mask=mask)

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, 8, cv2.CV_32S)

    valid_components = []
    # Start from label 1, as label 0 is the background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area_range[0] < area < area_range[1]:
            component_mask = (labels == i).astype("uint8") * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_components.append({
                'contour': contours[0],
                'centroid': centroids[i]
            })

    return valid_components, binary_img

def analyze_holes_in_region_v6(gray_image, region_info, config):
    """[V6 Core] Analyzes holes using the dual-threshold connected components method."""
    region_mask = region_info['mask']
    region_contour = region_info['contour']
    region_area = cv2.contourArea(region_contour)
    x, y, w, h = cv2.boundingRect(region_contour)

    # 1. Dynamically calculate the area range for hole filtering
    min_hole_area = region_area * config.HOLE_MIN_AREA_RATIO_OF_REGION
    max_hole_area = region_area * config.HOLE_MAX_AREA_RATIO_OF_REGION

    # 2. Determine thresholds
    uncovered_threshold = config.UNCOVERED_HOLE_THRESHOLD
    if config.ALL_HOLES_THRESHOLD == "auto":
        # Automatically calculate the high threshold: find the grayscale peak of the grid skeleton, then subtract an offset.
        hist = cv2.calcHist([gray_image], [0], region_mask, [256], [1, 256])
        peak_intensity = np.argmax(hist)
        all_holes_threshold = peak_intensity - config.ALL_HOLES_THRESHOLD_OFFSET
        print(f"   - Auto-determined high threshold: Grid peak={peak_intensity}, Calculated threshold={all_holes_threshold}")
    else:
        all_holes_threshold = config.ALL_HOLES_THRESHOLD

    # 3. Step One: Use the low threshold to precisely find "uncovered" black holes
    uncovered_components, uncovered_binary = get_components_from_threshold(
        gray_image, region_mask, uncovered_threshold, (min_hole_area, max_hole_area)
    )

    # 4. Step Two: Use the high threshold to find "all" holes
    all_components, all_binary = get_components_from_threshold(
        gray_image, region_mask, all_holes_threshold, (min_hole_area, max_hole_area)
    )

    # 5. Classify: Exclude "uncovered" holes from "all" holes; the remainder are "covered" holes
    covered_components = []
    uncovered_centroids = {tuple(comp['centroid']) for comp in uncovered_components}

    for all_comp in all_components:
        is_uncovered = False
        # Check if its centroid is very close to a known "uncovered" hole's centroid
        for uncov_cent in uncovered_centroids:
            dist = np.linalg.norm(np.array(all_comp['centroid']) - np.array(uncov_cent))
            if dist < 10: # 10-pixel tolerance
                is_uncovered = True
                break
        if not is_uncovered:
            covered_components.append(all_comp)

    # 6. Prepare the results for return
    analysis_results = {
        'covered': [c['contour'] for c in covered_components],
        'uncovered': [c['contour'] for c in uncovered_components],
        'stages': {
            "1. Original": gray_image[y:y+h, x:x+w],
            f"2. Uncovered (T<{uncovered_threshold})": uncovered_binary[y:y+h, x:x+w],
            f"3. All Holes (T<{all_holes_threshold})": all_binary[y:y+h, x:x+w],
        }
    }
    return analysis_results

# (segment_grid_cells and the main function remain largely unchanged, just need to call the V6 analysis function)
def segment_grid_cells(gray_image, original_bgr_image, config):
    # (This function's logic is identical to the previous version)
    print("Step 1: Starting segmentation of grid regions...")
    _, thresh_inverted = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh_inverted, cv2.MORPH_CLOSE, kernel, iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_regions = []
    output_image_with_contours = original_bgr_image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if config.REGION_MIN_AREA < area < config.REGION_MAX_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w == 0: continue
            aspect_ratio = float(w) / h
            if config.REGION_MIN_ASPECT_RATIO < aspect_ratio < config.REGION_MAX_ASPECT_RATIO:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                solidity = float(area) / hull_area
                if solidity > config.REGION_MIN_SOLIDITY:
                    mask = np.zeros_like(gray_image)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    valid_regions.append({'contour': contour, 'mask': mask})
                    cv2.drawContours(output_image_with_contours, [contour], -1, config.COLOR_REGION, 3)
    print(f"Found {len(contours)} initial contours, filtered to {len(valid_regions)} valid grid regions.")
    return output_image_with_contours, valid_regions

def main(image_path, output_dir, config):
    img = cv2.imread(image_path); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is None: print(f"Error: Could not read image '{image_path}'"); return
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    visual_image, regions = segment_grid_cells(gray, img, config)
    if not regions: print("No valid grid regions were found."); return

    total_stats = {'covered': 0, 'uncovered': 0}
    for i, region_info in enumerate(regions):
        print(f"\n--- Analyzing Region #{i+1} ---")

        # Call the V6 analysis function
        analysis_results = analyze_holes_in_region_v6(gray, region_info, config)

        num_covered = len(analysis_results['covered']); num_uncovered = len(analysis_results['uncovered'])
        total_stats['covered'] += num_covered; total_stats['uncovered'] += num_uncovered
        print(f"   - Covered Holes: {num_covered}, Uncovered Holes: {num_uncovered}")

        cv2.drawContours(visual_image, analysis_results['covered'], -1, config.COLOR_COVERED, -1)
        cv2.drawContours(visual_image, analysis_results['uncovered'], -1, config.COLOR_UNCOVERED, -1)

        # Generate and save the V6 diagnostic image
        title = f"Region #{i+1} | T_uncov={config.UNCOVERED_HOLE_THRESHOLD} | T_all={config.ALL_HOLES_THRESHOLD}"
        debug_image = create_debug_visualization_v6(analysis_results['stages'], analysis_results, title)
        debug_filename = os.path.join(output_dir, f"region_{i+1}_debug_stages.jpg")
        cv2.imwrite(debug_filename, debug_image)
        print(f"   - Detailed diagnostic image for region #{i+1} saved to: {debug_filename}")

    # (Summarize, display, and save the final result image)
    total_holes = total_stats['covered'] + total_stats['uncovered']
    if total_holes > 0:
        overall_coverage = total_stats['covered'] / total_holes * 100
        summary_text = f"Overall Coverage: {overall_coverage:.2f}% ({total_stats['covered']}/{total_holes})"
    else: summary_text = "No holes detected."
    print("\nFinal Statistics:", summary_text)

    cv2.putText(visual_image, summary_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(visual_image, summary_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    final_output_path = os.path.join(output_dir, "final_result.jpg")
    cv2.imwrite(final_output_path, visual_image)
    print(f"\nFinal result image saved to: '{final_output_path}'")
    cv2.imshow("Final Result (V6)", cv2.resize(visual_image, (1024, 1024))); cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    INPUT_IMAGE_PATH = "testdata/5-4_019.jpg"
    OUTPUT_DIR = "analysis_results"
    config = Config()

    if not os.path.exists(INPUT_IMAGE_PATH) or "path/to/your/image" in INPUT_IMAGE_PATH:
        print("="*60 + "\nError: Please set a valid image file path in the 'INPUT_IMAGE_PATH' variable.\n" + "="*60)
    else:
        main(INPUT_IMAGE_PATH, OUTPUT_DIR, config)