import os
from typing import Optional, Callable, TypeAlias

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms import v2 as transforms

class_names = (
    'Car',
    'Van',
    'Truck',
    'Pedestrian', 
    'Person_sitting',
    'Cyclist',
    'Tram',
    'Misc',
    'DontCare',
)

DataSample: TypeAlias = tuple[torch.Tensor, dict[str, torch.Tensor]]

class KittiDetectionDataset(Dataset):
    
    def __init__(self,
                 image_dir_path: str,
                 label_dir_path: str,
                 transform: Optional[Callable] = None):
        
        assert os.path.isdir(image_dir_path)
        assert os.path.isdir(label_dir_path)

        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.transform = transform

        self.n_samples = len(os.listdir(image_dir_path))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> DataSample:
        filename = f'{index:06d}'

        img = read_image(os.path.join(self.image_dir_path, filename + '.png'))
        labels, boxes = self._read_labels(os.path.join(self.label_dir_path, filename + '.txt'))

        target = {
            'labels': torch.tensor(labels, dtype=torch.int),
            'boxes': BoundingBoxes(boxes, format='XYXY', canvas_size=transforms.functional.get_size(img)),
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def _read_labels(self, path: str) -> tuple[list[int], list[tuple[float]]]:
        with open(path) as label_file:
            class_labels = []
            boxes = []
            for line in label_file.readlines():
                class_label, bbox = self._parse_object_line(line)
                class_labels.append(class_label)
                boxes.append(bbox)

        return class_labels, boxes

    def _parse_object_line(self, line: str) -> tuple[int, tuple[float]]:
        elements = line.split()

        object_class = class_names.index(elements[0])
        
        left, top, right, bottom = elements[4:8]
        bbox = (left, top, right, bottom)
        bbox = tuple(float(x) for x in bbox)
        
        return object_class, bbox
    