import json
from pathlib import Path
import attrs
from typing import Any
import numpy as np
from shapely import line_interpolate_point
from shapely.geometry import LinearRing
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.functional import resize


RESAMPLED_POLYGON_POINTS = 16
DETECTION_TARGET_IMAGE_SIZE = (1024, 1024)
RECOGNITION_TARGET_IMAGE_SIZE = (32, 128)
DATASET_DIR = Path("dataset/TextOCR")


@attrs.frozen
class DoctrDetSample:
    image_path: Path
    width: int
    height: int
    polygons: np.ndarray


def sanitize_polygon(
    points: np.ndarray,
    target_points: int = RESAMPLED_POLYGON_POINTS,
) -> np.ndarray:
    """Resample polygon vertices using Shapely's contour interpolation."""

    ring = LinearRing(points)
    perimeter = float(ring.length)
    distances = np.linspace(0.0, perimeter, num=target_points, endpoint=False, dtype=np.float32)

    samples = np.empty((target_points, 2), dtype=np.float32)
    for idx, dist in enumerate(distances):
        interp_pt = line_interpolate_point(ring, float(dist), normalized=False)
        samples[idx, 0] = float(interp_pt.x)
        samples[idx, 1] = float(interp_pt.y)

    return samples


class TextOCRDoctrDetDataset(Dataset[tuple[Tensor, dict[str, np.ndarray]]]):
    """Minimal TextOCR dataset wrapper emitting Doctr detection targets."""

    def __init__(
        self,
        images_dir: Path | None = None,
        json_path: Path | None = None,
        num_samples: int | None = None,
    ) -> None:
        base_images_dir = images_dir or (DATASET_DIR / "train_val_images")
        metadata_path = json_path or (DATASET_DIR / "TextOCR_0.1_train.json")

        with open(metadata_path) as fh:
            json_data = json.load(fh)

        self.samples: list[DoctrDetSample] = []
        max_labels, _ = self.get_max_dimensions(json_data)
        max_pts = RESAMPLED_POLYGON_POINTS

        for img_id, ann_ids in json_data["imgToAnns"].items():
            if num_samples is not None and len(self.samples) >= num_samples:
                break

            img_meta = json_data["imgs"][img_id]
            width, height = img_meta["width"], img_meta["height"]

            polygons: np.ndarray = np.zeros((max_labels, max_pts, 2), dtype=np.float32)
            for i, ann_id in enumerate(ann_ids):
                ann = json_data["anns"][ann_id]
                raw_points: list[float] = ann["points"]

                if not raw_points:
                    continue

                xs = np.array(raw_points[0::2], dtype=np.float32)
                ys = np.array(raw_points[1::2], dtype=np.float32)

                # Normalize
                xs /= width
                ys /= height

                # Clip to image
                xs = np.clip(xs, 0.0, float(DETECTION_TARGET_IMAGE_SIZE[0]))
                ys = np.clip(ys, 0.0, float(DETECTION_TARGET_IMAGE_SIZE[1]))

                stacked = np.stack([xs, ys], axis=1)
                cleaned = sanitize_polygon(stacked, target_points=max_pts)
                if cleaned is None:
                    continue
                polygons[i] = cleaned

            sample = DoctrDetSample(
                image_path=Path(base_images_dir) / img_meta["file_name"],
                width=width,
                height=height,
                polygons=polygons,
            )
            self.samples.append(sample)

        if not self.samples:
            raise ValueError(
                "No samples found. Ensure the TextOCR dataset is available at"
                f" {base_images_dir} and json at {metadata_path}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, np.ndarray]]:
        sample = self.samples[idx]

        image = decode_image(str(sample.image_path))
        image = resize(image, list(DETECTION_TARGET_IMAGE_SIZE), antialias=True)
        image = image.to(torch.float32) / 255.0

        # Doctr expects a dict keyed by class name containing relative boxes in float32 numpy.
        target = {"words": sample.polygons.copy()}
        return image, target

    def get_max_dimensions(self, json_data: Any) -> tuple[int, int]:
        max_labels = 0
        for ann_ids in json_data["imgToAnns"].values():
            max_labels = max(max_labels, len(ann_ids))

        max_pts = 0
        for ann in json_data["anns"].values():
            max_pts = max(max_pts, len(ann["points"]))

        return max_labels, max_pts


def doctr_detection_collate(
    batch: list[tuple[Tensor, dict[str, np.ndarray]]],
) -> tuple[Tensor, list[dict[str, np.ndarray]]]:
    # Overrides default numpy collate and keeps targets as numpy arrays as expected by doctr.

    images, targets = zip(*batch)
    stacked_images = torch.stack(images, dim=0)

    # Deep copy dicts for each sample in case doctr modifies them.
    collated_targets = [{k: v.copy() for k, v in target.items()} for target in targets]
    return stacked_images, collated_targets


import json
from pathlib import Path
import attrs
from typing import Any, Optional, Callable, cast, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_file, decode_image
from torchvision.transforms.functional import resize
from doctr.utils.geometry import extract_crops

@attrs.frozen
class DoctrRecSample:
    """
    A sample structure for text recognition, holding the path to an image,
    the polygon coordinates of a single word, the word's text label, and
    the original image dimensions.

    Args:
        image_path: Path to the full image file.
        points: A numpy array of shape (N, 2) representing the polygon of the word.
        label: The text string for the word.
        width: The original width of the full image.
        height: The original height of the full image.
    """
    image_path: Path
    points: np.ndarray
    label: str
    width: int
    height: int


class TextOCRDoctrRecDataset(Dataset[tuple[Tensor, str]]):
    """
    A PyTorch Dataset for the TextOCR dataset, specifically for text recognition tasks.
    Each item returned is a tuple containing a cropped image tensor of a single word
    and its corresponding text label.

    Args:
        images_dir: The directory where the dataset images are stored. If None, it
            defaults to DATASET_DIR / "train_val_images".
        json_path: The path to the TextOCR JSON annotation file. If None, it
            defaults to DATASET_DIR / "TextOCR_0.1_train.json".
        img_transforms: An optional callable to apply transformations to the
            cropped image tensor.
        num_samples: An optional integer to limit the number of samples used.
    """

    def __init__(
            self,
            images_dir: Optional[Path] = None,
            json_path: Optional[Path] = None,
            img_transforms: Optional[Callable[[Tensor], Tensor]] = None,
            num_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.img_transforms = img_transforms

        # Set default paths if none are provided
        base_images_dir = images_dir or (DATASET_DIR / "train_val_images")
        metadata_path = json_path or (DATASET_DIR / "TextOCR_0.1_train.json")

        if not base_images_dir.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Could not find dataset images or metadata. "
                f"Please ensure files are at {base_images_dir} and {metadata_path}"
            )

        with open(metadata_path) as f:
            json_data = json.load(f)

        self.samples = self._load_samples(json_data, base_images_dir)

        if num_samples:
            self.samples = self.samples[:num_samples]

    def _load_samples(self, json_data: Any, image_dir: Path) -> List[DoctrRecSample]:
        """Parses the JSON data to create a list of DoctrRecSample objects."""
        samples = []
        # Create a quick lookup for image metadata
        image_metadata = {img['id']: img for img in json_data["imgs"].values()}

        for ann in json_data["anns"].values():
            image_id = ann["image_id"]

            # Filter out annotations that are illegible, often marked with "."
            label = ann["utf8_string"]
            if label == "." or len(label) > 32:
                continue

            # Ensure the corresponding image metadata exists
            if image_id not in image_metadata:
                continue

            img_info = image_metadata[image_id]
            image_path = image_dir / img_info["file_name"]

            # The 'points' field provides polygon coordinates [x1, y1, x2, y2, ...]
            # Reshape them into an (N, 2) array of vertices.
            points = np.array(ann["points"], dtype=np.float32).reshape(-1, 2)

            samples.append(
                DoctrRecSample(
                    image_path=image_path,
                    points=points,
                    label=label,
                    width=img_info["width"],
                    height=img_info["height"],
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        """
        Fetches a single sample from the dataset.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing:
                - The cropped and transformed image tensor of the word.
                - The corresponding text label.
        """
        sample = self.samples[idx]

        try:
            # Load the full image using torchvision
            img_content = read_file(str(sample.image_path))
            full_image = decode_image(img_content, mode="RGB")

            # Resize and normalize the full image
            full_image = full_image.to(torch.float32) / 255.0

            # Calculate the bounding box from the polygon points
            xmin = int(np.floor(sample.points[:, 0].min()))
            ymin = int(np.floor(sample.points[:, 1].min()))
            xmax = int(np.ceil(sample.points[:, 0].max()))
            ymax = int(np.ceil(sample.points[:, 1].max()))

            xmin = np.clip(xmin, a_min=0, a_max=None)
            ymin = np.clip(ymin, a_min=0, a_max=None)

            cropped_image = full_image[:, ymin:ymax, xmin:xmax]
            cropped_image = resize(cropped_image, list(RECOGNITION_TARGET_IMAGE_SIZE), antialias=True)

            # Apply transformations if any are provided
            if self.img_transforms:
                cropped_image = self.img_transforms(cropped_image)

            return cropped_image, sample.label

        except Exception as e:
            # If an error occurs (e.g., file not found, cropping fails),
            # we'll try to load the next sample to prevent training from crashing.
            # A more robust solution could involve logging the error.
            print(f"Warning: Could not load or process sample {idx} ({sample.image_path}). Error: {e}. Skipping.")
            print(xmin, xmax, ymin, ymax)
            print(full_image.shape)
            return self.__getitem__((idx + 1) % len(self))


def doctr_recognition_collate(
        batch: List[tuple[Tensor, str]],
) -> tuple[Tensor, List[str]]:
    """
    Collate function for recognition model training. It overrides the default
    collate to return a list of targets instead of stacking them.

    Args:
        batch: A list of samples, where each sample is a tuple of an image tensor
            and its corresponding label.

    Returns:
        A tuple containing a stacked tensor of images and a list of labels.
    """
    #print((*batch)[0])
    images, labels = zip(*batch)
    return torch.stack(images, 0), list(labels)