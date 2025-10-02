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


TARGET_IMAGE_SIZE = (1024, 1024)
DATASET_DIR = Path("../../dataset/TextOCR")
RESAMPLED_POLYGON_POINTS = 16


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
                xs = np.clip(xs, 0.0, float(TARGET_IMAGE_SIZE[0]))
                ys = np.clip(ys, 0.0, float(TARGET_IMAGE_SIZE[1]))

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
        image = resize(image, list(TARGET_IMAGE_SIZE), antialias=True)
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
