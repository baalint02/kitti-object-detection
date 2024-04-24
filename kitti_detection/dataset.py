from kitti_detection import config

import os
from typing import Optional, Callable, TypeAlias

import torch
from torch.utils.data import Dataset, Subset, random_split
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
                 idx_range: Optional[range] = None,
                 transform: Optional[Callable] = None):
        
        assert os.path.isdir(image_dir_path)
        assert os.path.isdir(label_dir_path)

        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.indices = idx_range or range(0, len(os.listdir(image_dir_path)))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> DataSample:
        filename = f'{self.indices[index]:06d}'

        img = read_image(os.path.join(self.image_dir_path, filename + '.png'))
        labels, boxes = self._read_labels(os.path.join(self.label_dir_path, filename + '.txt'))

        target = {
            'labels': torch.tensor(labels, dtype=torch.int),
            'boxes': BoundingBoxes(boxes, format='XYXY', canvas_size=transforms.functional.get_size(img)),
        }
                
        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def _read_labels(self, path: str) -> tuple[list[int], list[tuple[float]]]:
        with open(path) as label_file:
            class_labels = []
            boxes = []
            for line in label_file.readlines():
                object_class, bbox = self._parse_object_line(line)

                if object_class == 'DontCare':
                    continue
                
                object_class = class_names.index(object_class)
                class_labels.append(object_class)
                boxes.append(bbox)

        return class_labels, boxes

    def _parse_object_line(self, line: str) -> tuple[int, tuple[float]]:
        elements = line.split()

        object_class = elements[0]
        
        left, top, right, bottom = elements[4:8]
        bbox = (left, top, right, bottom)
        bbox = tuple(float(x) for x in bbox)
        
        return object_class, bbox
    
def load_train_val_test_dataset(split=(0.7, 0.15, 0.15)) -> tuple[KittiDetectionDataset, KittiDetectionDataset, KittiDetectionDataset]:
    n_samples = len(os.listdir(config.DATA_IMAGE_DIR_PATH))
    train_end = round(n_samples * split[0])
    val_end = round(n_samples * (split[0] + split[1]))

    train = KittiDetectionDataset(config.DATA_IMAGE_DIR_PATH, config.DATA_LABEL_DIR_PATH, idx_range=range(0, train_end))
    val = KittiDetectionDataset(config.DATA_IMAGE_DIR_PATH, config.DATA_LABEL_DIR_PATH, idx_range=range(train_end,  val_end))
    test = KittiDetectionDataset(config.DATA_IMAGE_DIR_PATH, config.DATA_LABEL_DIR_PATH, idx_range=range(val_end,  n_samples))
    return train, val, test
