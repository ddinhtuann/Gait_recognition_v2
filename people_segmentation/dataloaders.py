from pathlib import Path
from typing import List, Dict, Any
from typing import Tuple

import albumentations as albu
import numpy as np
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb, load_grayscale
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        transform: albu.Compose,
        length: int = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % len(self.samples)

        image_path, mask_path = self.samples[idx]

        image = load_rgb(image_path, lib="cv2")
        mask = load_grayscale(mask_path)

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

        mask = (mask > 0).astype(np.uint8)

        mask = torch.from_numpy(mask)

        return {
            "image_id": image_path.stem,
            "features": tensor_from_rgb_image(image),
            "masks": torch.unsqueeze(mask, 0).float(),
        }


#
#
# class SegmentationDatasetTest(Dataset):
#     def __init__(self, file_names: List[Path], transform: albu.Compose) -> None:
#         self.file_names = file_names
#         self.transform = transform
#
#     def __len__(self) -> int:
#         return len(self.file_names)
#
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         image_path = self.file_names[idx]
#
#         image = load_rgb(image_path)
#
#         height, width = image.shape[:2]
#
#         # Resize
#         resizer = albu.Compose([albu.LongestMaxSize(max_size=1920, p=1)], p=1)
#         image = resizer(image=image)["image"]
#
#         # pad
#         image, pads = pad(image, factor=1920)
#
#         # apply augmentations
#         image = self.transform(image=image)["image"]
#
#         return {
#             "image_id": image_path.stem,
#             "features": tensor_from_rgb_image(image),
#             "pads": np.array(pads).T,
#             "height": height,
#             "width": width,
#         }
