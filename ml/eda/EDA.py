import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


# --- Utility Functions ---


def shoelace_area(points):
    """
    Calculates the area of a non-self-intersecting polygon given its vertices
    using the Shoelace formula. Points must be ordered (e.g., clockwise).
    The 'points' list is flat: [x1, y1, x2, y2, ...].
    """
    if len(points) < 6:
        return 0.0  # Not a valid polygon (needs at least 3 points/6 values)

    x = points[0::2]
    y = points[1::2]

    # Shoelace formula components
    sum1 = np.sum(np.array(x) * np.roll(np.array(y), -1))
    sum2 = np.sum(np.array(y) * np.roll(np.array(x), -1))

    return 0.5 * np.abs(sum1 - sum2)


def calculate_angle(p1, p2):
    """
    Calculates the rotation angle of a line segment (p1 to p2) with respect to the horizontal axis.
    p1 and p2 are (x, y) tuples. Returns angle in degrees (-90 to 90).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # np.arctan2 returns angle in radians (-pi to pi).
    # We use -np.degrees to get clockwise rotation from horizontal.
    # The TextOCR format defines points (x1, y1) to (x2, y2) as the top line of the text.
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # Adjust to typically expected OCR range (e.g., -90 to 90 or 0 to 180)
    # Since TextOCR uses clockwise order from top-left, we want the angle of the top edge
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg


# --- Main EDA Functions ---


def load_data(json_path):
    """Loads the TextOCR JSON file."""
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return None


def analyze_all_characteristics(data, output_folder):
    """
    Performs all requested EDA steps on the TextOCR dataset.
    """
    if not data:
        return

    # Extract all annotations for processing
    annotations = list(data.get("anns", {}).values())
    images = data.get("imgs", {})

    if not annotations or not images:
        print("Error: JSON data is missing 'anns' or 'imgs' keys.")
        return

    # 1. Bounding Box Point Number Distribution
    point_counts = Counter(len(ann["points"]) // 2 for ann in annotations)

    print("\n--- 1. Bounding Box Point Number Distribution ---")
    print(f"Total Annotations: {len(annotations)}")
    print("Point counts (number of vertices per polygon):")
    for count, freq in sorted(point_counts.items()):
        print(
            f"  {count} points (Quad/Poly): {freq} ({freq / len(annotations) * 100:.2f}%)"
        )

    # Visualization (Placeholder)
    plt.figure(figsize=(8, 5))
    plt.bar(point_counts.keys(), point_counts.values())
    plt.title("Distribution of Polygon Vertices (Points/2)")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Frequency (Word Annotations)")
    plt.savefig(os.path.join(output_folder, "point_distribution.png"))
    plt.close()
    print(
        f"Visualization saved to {os.path.join(output_folder, 'point_distribution.png')}"
    )

    # 2. Rotation and Skew Analysis
    rotation_angles = []
    skew_ratios = []

    for ann in annotations:
        points = ann["points"]
        bbox = ann["bbox"]  # [x, y, w, h]

        # Calculate Rotation Angle (of the top edge: p1 to p2)
        if len(points) >= 4:
            p1 = (points[0], points[1])
            p2 = (points[2], points[3])
            angle = calculate_angle(p1, p2)
            rotation_angles.append(angle)

        # Calculate Skew/Non-Rectangularity Ratio (Shoelace Area / Bbox Area)
        if (
            len(points) >= 8
        ):  # Only makes sense for quadrilaterals or more complex polygons
            polygon_area = shoelace_area(points)
            bbox_area = bbox[2] * bbox[3]  # w * h

            # Non-rectangularity Ratio: Polygon Area / Horizontal Bbox Area
            # Values significantly < 1.0 indicate substantial skew or empty space.
            # A perfect horizontal rectangle would yield a ratio of Area_poly / Area_poly
            # but here Area_bbox is the horizontal projection.
            if bbox_area > 0:
                # Use a simple ratio of the true polygonal area to the axis-aligned area
                skew_ratios.append(polygon_area / bbox_area)
            else:
                skew_ratios.append(0)

    print("\n--- 2. Rotation and Skew Analysis ---")
    print(f"Mean Rotation Angle (Top Edge): {np.mean(rotation_angles):.2f} degrees")
    print(
        f"Standard Deviation of Rotation Angle: {np.std(rotation_angles):.2f} degrees"
    )
    print(
        f"Mean Non-Rectangularity Ratio (Poly Area / BBox Area): {np.mean(skew_ratios):.4f}"
    )

    # Visualization (Rotation Histogram)
    plt.figure(figsize=(8, 5))
    plt.hist(rotation_angles, bins=50, range=[-90, 90])
    plt.title("Distribution of Text Rotation Angles (Top Edge)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency (Word Annotations)")
    plt.savefig(os.path.join(output_folder, "rotation_distribution.png"))
    plt.close()
    print(
        f"Visualization saved to {os.path.join(output_folder, 'rotation_distribution.png')}"
    )

    # 3. Text Variety (Printed vs. Handwritten)
    print("\n--- 3. Text Variety Analysis (Printed vs. Handwritten) ---")
    print(
        "Note: The standard TextOCR JSON schema typically does NOT include an explicit 'is_handwritten' field."
    )
    print("If available, it would usually be a separate field in the 'anns' object.")

    # Example if the field 'is_handwritten' existed:
    # handwritten_count = sum(1 for ann in annotations if ann.get('is_handwritten', False))
    # print(f"Handwritten Annotations: {handwritten_count}")

    # 4. Words per Image Distribution
    words_per_image = Counter(ann["image_id"] for ann in annotations)

    print("\n--- 4. Words per Image Distribution ---")
    print(f"Total Images: {len(images)}")
    print(
        f"Annotations per Image (Mean): {np.mean(list(words_per_image.values())):.2f}"
    )
    print(f"Annotations per Image (Max): {np.max(list(words_per_image.values()))}")

    # Visualization (Words per Image Histogram)
    plt.figure(figsize=(8, 5))
    plt.hist(words_per_image.values(), bins=50)
    plt.title("Distribution of Words (Annotations) Per Image")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency (Images)")
    plt.yscale("log")  # Use log scale for better visibility of high counts
    plt.savefig(os.path.join(output_folder, "words_per_image_distribution.png"))
    plt.close()
    print(
        f"Visualization saved to {os.path.join(output_folder, 'words_per_image_distribution.png')}"
    )

    # 5. Image Visualization of Text Location Heatmaps (Conceptual)
    print("\n--- 5. Image Visualization of Text Location Heatmaps ---")
    print(
        "Generating a heatmap requires merging annotations across all images, which is computationally intensive."
    )
    print("This placeholder logic prepares the data structure for plotting.")

    # Data aggregation for heatmap
    # Collect normalized coordinates (e.g., center points) for all annotations
    all_centers_normalized = []

    for ann in annotations:
        bbox = ann["bbox"]  # [x, y, w, h]
        image_id = ann["image_id"]

        # Get image dimensions to normalize coordinates
        img_meta = images.get(image_id)
        if not img_meta:
            continue

        img_w = img_meta["width"]
        img_h = img_meta["height"]

        # Calculate center point
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2

        # Normalize to (0, 1) range
        if img_w > 0 and img_h > 0:
            all_centers_normalized.append((center_x / img_w, center_y / img_h))

    # Conceptual Heatmap Plot (using 2D histogram of normalized coordinates)
    centers_x = [c[0] for c in all_centers_normalized]
    centers_y = [c[1] for c in all_centers_normalized]

    plt.figure(figsize=(8, 8))
    # Create 2D histogram (heatmap)
    plt.hist2d(
        centers_x, centers_y, bins=[50, 50], cmap="viridis", range=[[0, 1], [0, 1]]
    )
    plt.colorbar(label="Frequency of Word Centers")
    plt.title("Normalized Text Location Heatmap (Word Centers)")
    plt.xlabel("Normalized X Position (0=Left, 1=Right)")
    plt.ylabel("Normalized Y Position (0=Top, 1=Bottom)")
    plt.gca().invert_yaxis()  # Conventionally, Y=0 is the top of the image
    plt.savefig(os.path.join(output_folder, "text_location_heatmap.png"))
    plt.close()
    print(
        f"Visualization saved to {os.path.join(output_folder, 'text_location_heatmap.png')}"
    )


def get_characteristics(json_path, output_folder, image_dir=None):
    """
    Main function to orchestrate the EDA.
    image_dir is not used in this core EDA but is included for completeness.
    """
    # 1. Setup
    os.makedirs(output_folder, exist_ok=True)

    # 2. Get Data
    data = load_data(json_path)
    if not data:
        return

    # 3. Perform Analysis
    analyze_all_characteristics(data, output_folder)

    # 4. Final Summary
    print(f"\nEDA Complete. All results and visualizations saved in '{output_folder}'.")


if __name__ == "__main__":
    # Define your paths here based on your described directory structure:
    # EDA.py is in TextFlow, JSON is in TextFlow/TextOCR

    # Assuming EDA.py is run from the 'TextFlow' directory:
    TEXTOCR_JSON_PATH = os.path.join("dataset", "TextOCR", "TextOCR_0.1_val.json")

    TEXTOCR_IMAGE_DIR = os.path.join(
        "dataset", "TextOCR", "train_val_images", "train_images"
    )
    OUTPUT_FOLDER = os.path.join("dataset", "TextOCR", "EDA_Results")

    # Ensure the path reflects the uploaded file name
    # We will use the uploaded file's exact name and assume the JSON is local to TextFlow/TextOCR
    # For this environment, we rely on the file system paths provided by the user.

    # If the script is run from the root of the project:
    # TEXTOCR_JSON_PATH = "TextFlow/TextOCR/TextOCR_0.1_val.json"

    # Use the local path based on where the files were fetched:
    # I'll use a generic path that assumes the file is accessible in the relative structure.
    # NOTE: Since the file was uploaded, I must ensure the path is correct or passed as an argument.
    # For deployment flexibility, I'll update the path based on the user's description.

    # The user implies the file paths relative to where EDA.py resides:

    # Adjusting to use the uploaded file name directly, assuming the environment can find it.
    # In a real environment, you'd use the explicit path:
    TEXTOCR_JSON_PATH_LOCAL = "dataset/TextOCR/TextOCR_0.1_val.json"

    # Running with the necessary paths
    get_characteristics(
        json_path=TEXTOCR_JSON_PATH,
        output_folder=OUTPUT_FOLDER,
        image_dir=TEXTOCR_IMAGE_DIR,  # Not used in current EDA but included for consistency
    )
