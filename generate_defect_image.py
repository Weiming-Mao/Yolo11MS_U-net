import numpy as np
import cv2
import random
import json
import os

# Read the image
def read_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image file: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    return img

# Extract polygon annotations from a labelme JSON file
def load_labelme_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    polygons = []
    labels = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = shape['points']
            # Convert floating-point coordinates to integers
            polygon = np.array([[int(x), int(y)] for [x, y] in points], dtype=np.int32)
            polygons.append(polygon)
            labels.append(shape['label'])  # Extract the label
    return polygons, labels

# Select a random background image
def get_random_background(background_folder):
    background_files = [f for f in os.listdir(background_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    if not background_files:
        raise FileNotFoundError("No image files in the background folder")
    background_file = random.choice(background_files)
    background_img = cv2.imread(os.path.join(background_folder, background_file))
    if background_img is None:
        raise FileNotFoundError(f"Cannot read background image: {background_file}")
    background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    return background_img

# Resize the background image proportionally and pad with a white border
def resize_background(background_img, target_width, target_height):
    bg_height, bg_width = background_img.shape[:2]
    scale = min(target_width / bg_width, target_height / bg_height)  # Calculate the scaling ratio
    new_width = int(bg_width * scale)
    new_height = int(bg_height * scale)
    resized_bg = cv2.resize(background_img, (new_width, new_height))  # Proportional scaling

    # Create a white background of the target size
    padded_bg = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255  # White border
    x_offset = (target_width - new_width) // 2  # Horizontal offset
    y_offset = (target_height - new_height) // 2  # Vertical offset
    padded_bg[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_bg  # Fill with the scaled background
    return padded_bg

# Check if two polygons intersect
def polygons_intersect(polygon1, polygon2):
    # Use bounding box intersection as a fast check
    contour1 = np.array(polygon1, dtype=np.int32).reshape((-1, 1, 2))
    contour2 = np.array(polygon2, dtype=np.int32).reshape((-1, 1, 2))
    rect1 = cv2.boundingRect(contour1)
    rect2 = cv2.boundingRect(contour2)
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False  # Do not intersect
    return True  # Intersect

# Adjust the color and contrast of the image
def adjust_color_and_contrast(img, label):
    # Adjust brightness and contrast based on the label
    if label == "damage":
        alpha = random.uniform(0.7, 0.9)  # Decrease contrast
        beta = random.randint(-20, -10)   # Decrease brightness (make it darker)
    elif label == "stain":
        alpha = random.uniform(1.1, 1.3)  # Increase contrast
        beta = random.randint(20, 30)     # Increase brightness (make it whiter)
    else:
        alpha = random.uniform(0.8, 1.2)  # Default contrast
        beta = random.randint(-10, 10)    # Default brightness

    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

# Transform the polygon and the image within it
def transform_polygon_and_image(polygon, img, operation_type='rotate', label=None):
    if not isinstance(polygon, np.ndarray) or polygon.ndim != 2 or polygon.shape[1] != 2:
        print(f"Invalid polygon: {polygon}")
        return img, polygon

    # Calculate the center of the polygon
    center = np.mean(polygon, axis=0)
    center = tuple(np.round(center).astype(int))
    center = (int(center[0]), int(center[1]))

    if operation_type == 'rotate':
        angle = random.randint(0, 360)
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        transformed_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        transformed_polygon = cv2.transform(np.array([polygon], dtype=np.float32), matrix)[0]
    elif operation_type == 'scale':
        # Adjust the scaling factor based on the label
        if label == "damage":
            scale = random.uniform(0.7, 0.9)  # Scale down
        elif label == "stain":
            scale = random.uniform(1.2, 1.4)  # Scale up
        else:
            scale = random.uniform(0.8, 1.2)  # Default scale
        matrix = cv2.getRotationMatrix2D(center, 0, scale)
        transformed_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        transformed_polygon = cv2.transform(np.array([polygon], dtype=np.float32), matrix)[0]
    elif operation_type == 'flip':
        flip_code = random.choice([-1, 0, 1])
        transformed_img = cv2.flip(img, flip_code)
        transformed_polygon = np.array([[img.shape[1] - x if flip_code == 1 else x,
                                         img.shape[0] - y if flip_code == 0 else y]
                                        for [x, y] in polygon], dtype=np.int32)
    elif operation_type == 'translate':
        dx = random.randint(-50, 50)
        dy = random.randint(-50, 50)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        transformed_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        transformed_polygon = cv2.transform(np.array([polygon], dtype=np.float32), matrix)[0]
    else:
        transformed_img, transformed_polygon = img, polygon

    transformed_polygon = transformed_polygon.astype(np.int32)
    return transformed_img, transformed_polygon

def random_crop_polygon(polygon, img, label):
    # Get the height and width of the image
    img_height, img_width = img.shape[:2]
    
    # Create a mask of the original polygon
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], (255, 255, 255))  # Fill the original polygon area
    
    # Get the bounding rectangle of the original polygon
    x, y, w, h = cv2.boundingRect(polygon)
    
    # Limit the bounding rectangle to ensure it does not exceed image boundaries
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    # Only perform random cropping for polygons with the "damage" label
    if label == "damage" and random.random() < 0.5:  # 50% chance to perform the crop
        # Randomly generate a convex quadrilateral, pentagon, hexagon, or heptagon
        num_vertices = random.choice([4, 5, 6, 7])  # Randomly select the number of vertices
        new_polygon = []
        attempts = 0
        max_attempts = 100  # Maximum number of attempts to avoid an infinite loop
        
        while len(new_polygon) < num_vertices and attempts < max_attempts:
            # Generate a random point within the bounding rectangle
            px = random.randint(x, x + w)
            py = random.randint(y, y + h)
            
            # Ensure the point's coordinates are within the image boundaries
            if px >= img_width or py >= img_height:
                attempts += 1
                continue
            
            # Check if the point is inside the original polygon
            if mask[py, px].sum() > 0:  # Point is inside the polygon
                new_polygon.append([px, py])
            attempts += 1
        
        if len(new_polygon) < num_vertices:
            # If enough vertices cannot be generated, return the original image and polygon
            return img, polygon.reshape((-1, 1, 2))
        
        # Ensure the polygon is convex
        new_polygon = np.array(new_polygon, dtype=np.int32)
        hull = cv2.convexHull(new_polygon)  # Calculate the convex hull
        new_polygon = hull.reshape((-1, 1, 2))  # Convert to (N, 1, 2) shape
        
        # Create a mask to extract the new polygon area
        new_mask = np.zeros_like(img, dtype=np.uint8)
        cv2.fillPoly(new_mask, [new_polygon], (255, 255, 255))
        cropped_img = cv2.bitwise_and(img, new_mask)
        
        return cropped_img, new_polygon
    else:
        # If not cropping, return the original image and polygon directly
        return img, polygon.reshape((-1, 1, 2))  # Ensure the shape is (N, 1, 2)

def overlay_on_image(background_folder, transformed_polygon, transformed_img, original_polygon, original_img, label):
    # Get a random background image
    background_img = get_random_background(background_folder)
    
    # Resize the background image to match the original image dimensions
    background_img = resize_background(background_img, original_img.shape[1], original_img.shape[0])
    
    # Randomly crop the polygon area (only for the "damage" label)
    cropped_img, cropped_polygon = random_crop_polygon(transformed_polygon, transformed_img, label)
    
    # Calculate the center point of the background image
    bg_center_x = background_img.shape[1] // 2  # Center of the background image width
    bg_center_y = background_img.shape[0] // 2  # Center of the background image height
    
    # Calculate the radius: take the smaller of the background width/height and multiply by 0.15
    radius = int(min(background_img.shape[1], background_img.shape[0]) * 0.15)
    
    # Calculate the center point of the current polygon area
    polygon_center = np.mean(cropped_polygon, axis=0).astype(int)
    if polygon_center.ndim == 1:  # If the result is a 1D array, convert it to 2D
        polygon_center = polygon_center.reshape(1, -1)
    polygon_center_x, polygon_center_y = polygon_center[0][0], polygon_center[0][1]
    
    # Calculate the translation amount to move the polygon center within the radius of the background center
    dx = bg_center_x - polygon_center_x  # Horizontal translation amount
    dy = bg_center_y - polygon_center_y  # Vertical translation amount
    
    # Limit the translation amount to be within the radius
    distance = np.sqrt(dx**2 + dy**2)  # Calculate the distance from the current center to the background center
    if distance > radius:
        scale = radius / distance  # Calculate the scaling ratio
        dx = int(dx * scale)  # Scale the horizontal translation amount proportionally
        dy = int(dy * scale)  # Scale the vertical translation amount proportionally
    
    # Translate the polygon area
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])  # Affine transformation matrix
    translated_img = cv2.warpAffine(cropped_img, matrix, (background_img.shape[1], background_img.shape[0]))
    
    # Translate the polygon vertices
    translated_polygon = cv2.transform(cropped_polygon.reshape(1, -1, 2), matrix).reshape((-1, 1, 2)).astype(np.int32)
    
    # Create a mask
    mask = np.zeros_like(background_img, dtype=np.uint8)
    cv2.fillPoly(mask, [translated_polygon], (255, 255, 255))  # Fill the polygon area
    
    # Overlay the cropped region onto the corresponding position of the background image
    for i in range(background_img.shape[0]):
        for j in range(background_img.shape[1]):
            if mask[i, j].sum() > 0:  # If the current pixel is within the polygon area
                background_img[i, j] = translated_img[i, j]  # Cover the background with the cropped image
    
    return background_img, translated_polygon

def generate_labelme_json(new_img, transformed_polygon, save_path, label="damage"):
    height, width, _ = new_img.shape
    
    # Reshape transformed_polygon from (N, 1, 2) to (N, 2)
    if transformed_polygon.ndim == 3 and transformed_polygon.shape[1] == 1:
        transformed_polygon = transformed_polygon.reshape(-1, 2)
    
    # Build the shapes list
    shapes = [
        {
            "label": label,  # Set the label name
            "points": transformed_polygon.tolist(),  # Polygon vertices
            "group_id": None,  # Group ID
            "description": "",  # Description
            "shape_type": "polygon",  # Shape type
            "flags": {}  # Flags
            # Remove the "mask" field, as LabelMe might not support it
        }
    ]
    
    # Build the labelme_data dictionary
    labelme_data = {
        "version": "5.4.1",  # Version number
        "flags": {},  # Global flags
        "shapes": shapes,  # List of shapes
        "imagePath": os.path.basename(save_path).replace('.json', '.jpg'),  # Image file name
        "imageData": None,  # Image data (optional)
        "imageHeight": height,  # Image height
        "imageWidth": width  # Image width
    }
    
    # Save the generated JSON file
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(
            labelme_data,
            f,
            indent=2,  # Indent with 2 spaces
            ensure_ascii=False  # Ensure non-ASCII characters are displayed correctly
        )
        f.write("\n")  # Add a newline at the end of the file

# Main function: process all images in a folder
def process_image_folder(image_folder, labelme_folder, background_folder, output_image_folder, output_json_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        labelme_json_path = os.path.join(labelme_folder, os.path.splitext(image_file)[0] + '.json')
        if not os.path.exists(labelme_json_path):
            print(f"Skipping: Corresponding annotation file not found: {labelme_json_path}")
            continue
        try:
            original_img = read_image(image_path)
            polygons, labels = load_labelme_json(labelme_json_path)
            if not polygons:
                print(f"Skipping: No valid polygon annotations found in: {image_file}")
                continue

            # Randomly select the number of transformations (at least one)
            num_transforms = random.randint(1, len(polygons))
            selected_indices = random.sample(range(len(polygons)), num_transforms)
            selected_polygons = [polygons[i] for i in selected_indices]
            selected_labels = [labels[i] for i in selected_indices]

            # Randomly duplicate polygons (up to 3 copies)
            max_copies = 3
            copied_polygons = []
            copied_labels = []
            for polygon, label in zip(selected_polygons, selected_labels):
                num_copies = random.randint(1, max_copies)
                for _ in range(num_copies):
                    copied_polygons.append(polygon)
                    copied_labels.append(label)

            # Transform each polygon
            for idx, (polygon, label) in enumerate(zip(copied_polygons, copied_labels)):
                # Randomly select a transformation operation
                operation_type = random.choice(['rotate', 'scale', 'flip', 'translate'])
                transformed_img, transformed_polygon = transform_polygon_and_image(polygon, original_img, operation_type, label)

                # Adjust color and contrast
                transformed_img = adjust_color_and_contrast(transformed_img, label)

                # Composite the image
                new_img, new_polygon = overlay_on_image(background_folder, transformed_polygon, transformed_img, polygon, original_img, label)

                # Save the new image
                output_image_name = f"Aaugmented_{os.path.splitext(image_file)[0]}_{idx}.jpg"
                output_image_path = os.path.join(output_image_folder, output_image_name)
                cv2.imwrite(output_image_path, cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))

                # Save the annotation file
                output_json_name = f"Aaugmented_{os.path.splitext(image_file)[0]}_{idx}.json"
                output_json_path = os.path.join(output_json_folder, output_json_name)
                generate_labelme_json(new_img, new_polygon, output_json_path, label=label)
                print(f"Processing complete: {output_image_name}, {output_json_name}")
        except Exception as e:
            print(f"Processing failed for: {image_file}, Error: {e}")

# Call the main function
image_folder = "C:/Users/admin/Desktop/defect/jpg"  # Input image folder path
labelme_folder =  "C:/Users/admin/Desktop/defect/json"   # Input annotation folder path
background_folder = "C:/Users/admin/Desktop/defect/background"  # Replace with the background image folder path
output_image_folder = "C:/Users/admin/Desktop/modeltest/q/generjpg"  # Replace with the output image folder path
output_json_folder = "C:/Users/admin/Desktop/modeltest/q/generjson" # Replace with the output json folder path

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_json_folder, exist_ok=True)

process_image_folder(image_folder, labelme_folder, background_folder, output_image_folder, output_json_folder)