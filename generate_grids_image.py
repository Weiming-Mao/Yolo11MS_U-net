from PIL import Image, ImageDraw
import numpy as np
import random
import math
import json
import os
import sys
import time
import multiprocessing

# Set the target folder path
output_folder = "generateimage"
os.makedirs(output_folder, exist_ok=True)

# Total number of generations
num_images_to_generate = 300
# Image size
width, height = 500, 500

# Function to generate an image and its corresponding JSON
def generate_image_and_json(index):
    try:
        # Simulate the logic for drawing an image and generating JSON
        time.sleep(random.uniform(1, 7))  # Simulate a time-consuming operation
        print(f"Image {index} generated successfully.")
        # Randomly generate the number of squares
        num_squares = random.randint(1, 6)
        # Size of the squares
        square_size = 500  # Side length of the square
        # Define the grayscale level ranges
        background_gray_range = (190, 220)  # Background grayscale range
        square_gray_range = (135, 160)      # Square grayscale range
        circle_light_gray_range = (85, 115) # Light gray circle/hole range
        circle_dark_gray_range = (20, 60)   # Dark gray circle/hole range

        # Initialize the image
        image = Image.new("L", (width, height), background_gray_range[0])  # Default background gray value
        draw = ImageDraw.Draw(image)

        # Add background noise
        background = np.full((height, width), random.randint(background_gray_range[0], background_gray_range[1]), dtype=np.uint8)
        noise = np.random.randint(-10, 10, (height, width), dtype=np.int32)  # Random noise
        background = np.clip(background + noise, background_gray_range[0], background_gray_range[1])  # Apply noise

        # Update the image background
        image = Image.fromarray(background.astype(np.uint8))
        draw = ImageDraw.Draw(image)
        # Small circle/hole parameters
        circle_radius = 10  # Circle/hole radius

        # Generate a randomly rotated grid
        def generate_rotated_grid_with_padding(square_x0, square_y0, square_size, num_circles, angle_deg):
            angle_rad = math.radians(angle_deg)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

            # Dynamically calculate the extended range (to ensure full coverage and avoid blank areas)
            grid_spacing = square_size / math.sqrt(num_circles)
            padding_factor = 1.5  # Slightly enlarge the extended range to avoid blank corners
            padding = int(grid_spacing * max(abs(cos_a), abs(sin_a)) * padding_factor)

            # Extend the square boundaries
            extended_x0, extended_y0 = square_x0 - padding, square_y0 - padding
            extended_x1, extended_y1 = square_x0 + square_size + padding, square_y0 + square_size + padding

            # Center point
            center_x, center_y = (square_x0 + square_x1) // 2, (square_y0 + square_y1) // 2

            # Generate an unrotated grid (covering the extended area)
            grid_points = []
            start_x = extended_x0
            start_y = extended_y0
            num_rows = int((extended_y1 - extended_y0) // grid_spacing) + 1
            num_cols = int((extended_x1 - extended_x0) // grid_spacing) + 1

            for i in range(num_rows):
                for j in range(num_cols):
                    x = start_x + j * grid_spacing
                    y = start_y + i * grid_spacing

                    # Rotate the point
                    rotated_x = center_x + (x - center_x) * cos_a - (y - center_y) * sin_a
                    rotated_y = center_y + (x - center_x) * sin_a + (y - center_y) * cos_a

                    # Only keep points that are inside the original square
                    if square_x0 <= rotated_x <= square_x1 and square_y0 <= rotated_y <= square_y1:
                        grid_points.append((rotated_x, rotated_y))

            return grid_points

        # A list to record the type and bounding box coordinates for each circle/hole
        lst = []

        # Randomly select the positions for dark black circles/holes
        dark_circle_ratio = np.random.uniform(0.05, 0.1)
        num_circles = 200  # Total number of circles/holes
        num_dark_circles = int(num_circles * dark_circle_ratio)
        dark_circle_positions = np.random.choice(range(num_circles), num_dark_circles, replace=False)

        # Randomly select 5% of the circles/holes for shape distortion
        num_distorted_circles = int(0.05 * num_circles)  # 5% of small circles will be distorted
        distorted_circle_indices = np.random.choice(len(range(num_circles)), num_distorted_circles, replace=False)

        # A list to store the positions of already drawn squares
        squares = []

        # Ensure squares do not overlap
        def check_square_overlap(x0, y0, size):
            for square in squares:
                if not (x0 + size <= square[0] or x0 >= square[0] + size or y0 + size <= square[1] or y0 >= square[1] + size):
                    return True  # If they overlap
            return False

        # Generate multiple squares, and randomly distribute small circles/holes within each square
        for _ in range(num_squares):
            square_x0 = random.randint(0, width - square_size)
            square_y0 = random.randint(0, height - square_size)

            # Ensure squares do not overlap
            while check_square_overlap(square_x0, square_y0, square_size):
                square_x0 = random.randint(0, width - square_size)
                square_y0 = random.randint(0, height - square_size)

            square_x1 = square_x0 + square_size
            square_y1 = square_y0 + square_size

            # Randomly select a gray value for the square area
            square_gray = random.randint(square_gray_range[0], square_gray_range[1])

            # Draw the square
            draw.rectangle([square_x0, square_y0, square_x1, square_y1], fill=square_gray)

            # Rotation angle
            angle_deg = np.random.uniform(0, 360)
            positions = generate_rotated_grid_with_padding(square_x0, square_y0, square_size, num_circles, angle_deg)

            # Randomly select the positions for dark black circles/holes
            dark_circle_positions = np.random.choice(len(positions), num_dark_circles, replace=False)

            # Draw the small circles/holes
            for idx, (x, y) in enumerate(positions):
                # Determine if this circle/hole needs to be distorted
                if idx in distorted_circle_indices:
                    # Slightly adjust the circle's shape: small variations on the edge, keeping it generally circular
                    offset = np.random.uniform(-1, 1, size=(8, 2))  # Small random offsets
                    circle_points = [(x + circle_radius * np.cos(theta) + offset[i][0], 
                                      y + circle_radius * np.sin(theta) + offset[i][1]) for i, theta in enumerate(np.linspace(0, 2 * np.pi, 8))]
                    # Draw the irregular circle/hole
                    circle_gray = random.randint(circle_light_gray_range[0], circle_light_gray_range[1]) if idx not in dark_circle_positions else random.randint(circle_dark_gray_range[0], circle_dark_gray_range[1])
                    draw.polygon(circle_points, fill=circle_gray)
                else:
                    # If not distorted, keep it as a standard circle
                    circle_gray = random.randint(circle_light_gray_range[0], circle_light_gray_range[1]) if idx not in dark_circle_positions else random.randint(circle_dark_gray_range[0], circle_dark_gray_range[1])
                    draw.ellipse([x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius], 
                                  fill=circle_gray)

                # Calculate the bounding box and record the circle/hole type
                min_x = min(pt[0] for pt in circle_points) if idx in distorted_circle_indices else x - circle_radius
                max_x = max(pt[0] for pt in circle_points) if idx in distorted_circle_indices else x + circle_radius
                min_y = min(pt[1] for pt in circle_points) if idx in distorted_circle_indices else y - circle_radius
                max_y = max(pt[1] for pt in circle_points) if idx in distorted_circle_indices else y + circle_radius

                # If the circle/hole partially extends beyond the square's boundary, classify it as "edge"
                if min_x < square_x0 or max_x > square_x1 or min_y < square_y0 or max_y > square_y1:
                    circle_type = "edge"
                    # Ensure to only record the part of the bounding box that is inside the square
                    min_x = max(min_x, square_x0)
                    max_x = min(max_x, square_x1)
                    min_y = max(min_y, square_y0)
                    max_y = min(max_y, square_y1)
                
                # Record the circle/hole's type and its bounding box
                #if idx in distorted_circle_indices:
                    #circle_type = "distorted"
                elif idx in dark_circle_positions:
                    circle_type = "uncovered"
                else:
                    circle_type = "covered"

                # Record the bounding box of the circle/hole
                lst.append({
                    "circle_type": circle_type,
                    "min_x": min_x,
                    "max_x": max_x,
                    "min_y": min_y,
                    "max_y": max_y
                })

            # Add the square's coordinates to the list of drawn squares
            squares.append((square_x0, square_y0, square_x1, square_y1))

        # After drawing all squares and small circles/holes, convert to a NumPy format full image
        full_image = np.array(image)
        # Custom overlap area threshold (bounding box area must be > 2/3 of the circumscribed square's area)
        overlap_threshold = 2/3  # Default set to 2/3, can be adjusted

        # Get the grayscale value at the center of the bounding box
        def get_center_gray(image, circle_min_x, circle_max_x, circle_min_y, circle_max_y):
            center_x = int((circle_min_x + circle_max_x) / 2)
            center_y = int((circle_min_y + circle_max_y) / 2)
            return image[center_y, center_x]  # Extract the center point's grayscale value

        # Update the classification for circles/holes of type "edge"
        for i in range(len(lst)):
            item = lst[i]
            if item["circle_type"] == "edge":
                # Get information about the current circle/hole
                circle_min_x = item["min_x"]
                circle_max_x = item["max_x"]
                circle_min_y = item["min_y"]
                circle_max_y = item["max_y"]

                # Calculate the area of the bounding box
                bounding_box_area = (circle_max_x - circle_min_x) * (circle_max_y - circle_min_y)

                # Calculate the area of the circumscribed square (side length is 2 * circle_radius)
                square_area = (2 * circle_radius) ** 2

                # If the bounding box area is >= the overlap threshold, reclassify based on grayscale value
                if bounding_box_area / square_area >= overlap_threshold:
                    # Get the grayscale value at the center of the bounding box
                    center_gray = get_center_gray(full_image, circle_min_x, circle_max_x, circle_min_y, circle_max_y)

                    # Determine the class based on the grayscale value
                    if circle_light_gray_range[0] <= center_gray <= circle_light_gray_range[1]:  # Light gray circle/hole
                        lst[i]["circle_type"] = "covered"
                    elif circle_dark_gray_range[0] <= center_gray <= circle_dark_gray_range[1]:  # Dark gray circle/hole
                        lst[i]["circle_type"] = "uncovered"
                    else:
                        print(f"Warning: Gray value {center_gray} not within any defined range.")

        # Set the scaling factor for expansion
        scale_factor = 1.1  # Expand to 1.1 times the original bounding box size

        # Image dimensions (to ensure no out-of-bounds access)
        image_width, image_height = background.shape[1], background.shape[0]

        # Update the bounding boxes in the lst
        for item in lst:
            # Original bounding box coordinates
            min_x, max_x = item["min_x"], item["max_x"]
            min_y, max_y = item["min_y"], item["max_y"]
            
            # Calculate the increase in width and height
            delta_x = ((max_x - min_x) * (scale_factor - 1)) / 2
            delta_y = ((max_y - min_y) * (scale_factor - 1)) / 2

            # Update the coordinates
            new_min_x = max(0, min_x - delta_x)  # Ensure it is not less than 0
            new_min_y = max(0, min_y - delta_y)  # Ensure it is not less than 0
            new_max_x = min(image_width, max_x + delta_x)  # Ensure it does not exceed the image width
            new_max_y = min(image_height, max_y + delta_y)  # Ensure it does not exceed the image height

            # Replace with the expanded coordinates
            item["min_x"] = new_min_x
            item["min_y"] = new_min_y
            item["max_x"] = new_max_x
            item["max_y"] = new_max_y

        # Generate randomly sized noise circles/holes on the background, ensuring they do not overlap with any squares
        num_noise_circles = random.randint(10, 100)  # Number of noise circles/holes

        for _ in range(num_noise_circles):
            # Randomly generate the radius, position, and gray value for the noise circle/hole
            radius = random.randint(10, 30)  # Circle/hole radius
            x = random.randint(0, width - radius * 2)  # Random x-coordinate
            y = random.randint(0, height - radius * 2)  # Random y-coordinate

            # Randomly select a gray value for the noise circle/hole (in the range 225 to 255)
            circle_gray = random.randint(225, 255)

            # Check if this circle/hole overlaps with any square region
            overlap = False
            for square in squares:
                square_x0, square_y0, square_x1, square_y1 = square
                # Check if the circle's entire bounding box intersects with the square
                if (x + radius * 2 > square_x0 and x < square_x1 and y + radius * 2 > square_y0 and y < square_y1):
                    overlap = True
                    break

            # If there is no overlap, draw this noise circle/hole
            if not overlap:
                draw.ellipse([x, y, x + radius * 2, y + radius * 2], fill=circle_gray)
        # Dynamically generate filenames
        image_filename = os.path.join(output_folder, f"newimage_{index}.jpg")
        json_filename = os.path.join(output_folder, f"newimage_{index}.json")
    # Save the image
        image.save(image_filename,format="JPEG")
        print(f"Image saved as '{image_filename}'")

    # Generate JSON data
        output_data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],
            "imagePath": f"newimage_{index}.jpg",
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        # Add circle/hole data to the JSON
        for record in lst:
            shape = {
                "label": record["circle_type"],
                "points": [
                    [record["min_x"], record["min_y"]],
                    [record["max_x"], record["max_y"]]
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            output_data["shapes"].append(shape)

        # Save the JSON file
        with open(json_filename, "w") as json_file:
            json.dump(output_data, json_file, indent=4)
        print(f"Label data saved as '{json_filename}'")
    
    except Exception as e:
        print(f"Error during generation for image {index}: {e}")

# Define a function with timeout control
def generate_with_timeout(index, timeout=5):
    task_process = multiprocessing.Process(target=generate_image_and_json, args=(index,))
    task_process.start()

    # Wait for the subprocess to complete, with a timeout
    task_process.join(timeout)

    # Check for timeout
    if task_process.is_alive():
        print(f"Task {index} exceeded {timeout} seconds, skipping this task.")
        task_process.terminate()  # Forcefully terminate the subprocess
        task_process.join()  # Ensure the subprocess resources are released
    else:
        print(f"Task {index} completed within {timeout} seconds.")

# Main logic
if __name__ == "__main__":
    for i in range(1, num_images_to_generate + 1):
        print(f"Starting task {i}...")
        generate_with_timeout(i)

    print("All tasks are completed. Exiting the program.")
    sys.exit(0)  # Ensure the program exits automatically