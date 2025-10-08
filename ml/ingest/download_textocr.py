import argparse
import os
import zipfile

import requests
from tqdm import tqdm


# Default base directory for the TextOCR content (will be created)
DEFAULT_DATASET_BASE_DIR = os.path.join("dataset", "TextOCR")

# Official Download Links (TextOCR v0.1)
DOWNLOAD_URLS = {
    "images": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    "train_json": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
    "val_json": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json",
}


def download_file(url: str, local_path: str) -> bool:
    """Downloads a file from a URL to a local path."""
    print(f"-> Downloading: {os.path.basename(local_path)}...")
    try:
        # Stream download for large files
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        # Update progress indicator for large files
                        progress_bar.update(len(chunk))
                        f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] Failed to download {os.path.basename(local_path)}: {e}")
        return False
    return True


def extract_zip(zip_path, extract_dir):
    """Extracts a zip file to the specified directory."""
    print(f"-> Extracting {os.path.basename(zip_path)} to {extract_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("   ...Extraction complete.")
    except Exception as e:
        print(f"   [ERROR] Failed to extract {os.path.basename(zip_path)}: {e}")
        return False
    return True


# --- Main Logic ---


def setup_and_download_textocr(dataset_base_dir: str) -> None:
    """Checks for existing files and downloads/extracts missing components."""

    # 1. Create necessary directories
    image_extract_dir = os.path.join(dataset_base_dir, "train_val_images")
    json_dir = dataset_base_dir
    os.makedirs(image_extract_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    print(f"Ensured directories exist under: {dataset_base_dir}")
    print("--------------------------------------------------")

    # 2. Download Images
    images_zip_name = "train_val_images.zip"
    images_zip_path = os.path.join(dataset_base_dir, images_zip_name)

    # TextOCR images are in a single zip, which extracts into a 'train_val_images' folder,
    # but the JSON paths expect the images in 'train_val_images/train_images'.
    # The official zip extracts into a folder structure that includes a 'train' directory.
    # We target the parent directory of the final images folder for extraction.
    target_image_folder = os.path.join(
        image_extract_dir, "train"
    )  # Assuming the zip extracts a 'train' folder
    if (
        os.path.isdir(target_image_folder)
        and len(os.listdir(target_image_folder)) > 1000
    ):
        print(
            f"✅ Images appear to be downloaded and extracted in: {target_image_folder}"
        )
    else:
        # Download
        if download_file(DOWNLOAD_URLS["images"], images_zip_path):
            # Extract
            # Note: The 'train_val_images.zip' contains a structure that leads to the final images.
            # We extract it to IMAGE_DIR.
            if extract_zip(images_zip_path, image_extract_dir):
                # Clean up the zip file (optional)
                os.remove(images_zip_path)
                print(f"   Removed temporary zip file: {images_zip_name}")

    print("--------------------------------------------------")

    # 3. Download JSON Annotations
    json_files = {
        "train_json": "TextOCR_0.1_train.json",
        "val_json": "TextOCR_0.1_val.json",
    }

    for key, filename in json_files.items():
        json_path = os.path.join(json_dir, filename)

        if os.path.exists(json_path):
            print(f"✅ JSON file already exists: {filename}")
            continue

        download_file(DOWNLOAD_URLS[key], json_path)

    print("--------------------------------------------------")
    print("Dataset download and setup complete.")
    print(f"Data is organized in the '{dataset_base_dir}' folder.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare the TextOCR dataset."
    )
    parser.add_argument(
        "--dataset-base-dir",
        default=DEFAULT_DATASET_BASE_DIR,
        help="Destination directory for the dataset contents.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_and_download_textocr(dataset_base_dir=args.dataset_base_dir)
