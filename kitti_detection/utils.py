from kitti_detection.dataset import DataSample, class_names

import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

denormalize = v2.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])


def display_samples_v(samples):
    plt.rcParams["savefig.bbox"] = 'tight'

    bb_colors = ['blue', 'yellow', 'green', 'red', 'orange'] * 12
    
    if not isinstance(samples, tuple):
        samples = (samples)

    _, axs = plt.subplots(nrows=len(samples), squeeze=False, figsize=(12, 3.3 * len(samples)))

    plt.tight_layout()
    for i, sample in enumerate(samples):
        img, target = sample
        labels = [ class_names[label.item() - 1] for label in target['labels'] ]

        img = img.detach()
        img = img.to(torch.uint8)
        img = draw_bounding_boxes(img, boxes=target['boxes'], labels=labels, colors=bb_colors, width=3)
        img = F.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.show()

def display_samples_h(samples):
    plt.rcParams["savefig.bbox"] = 'tight'

    bb_colors = ['blue', 'yellow', 'green', 'red', 'orange'] * 16
    
    if not isinstance(samples, tuple):
        samples = (samples)

    _, axs = plt.subplots(ncols=len(samples), squeeze=False, figsize=(3.3 * len(samples), 12))

    plt.tight_layout()
    for i, sample in enumerate(samples):
        img, target = sample
        labels = [ class_names[label.item()] for label in target['labels'] ]

        img = img.detach()
        img = denormalize(img)
        img = img * 255
        img = img.to(torch.uint8)
        
        if target['boxes'].ndim == 1:
            print(f"Target box with wrong dimensionality: {target['boxes']}")

        else:
            img = draw_bounding_boxes(img, boxes=target['boxes'], labels=labels, colors=bb_colors, width=3)
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.show()


def display_one(img: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor):
    bb_colors = ['blue', 'yellow', 'green', 'red', 'orange'] * math.ceil(boxes.shape[0] / 5)

    if not labels.shape[0] == boxes.shape[0]:
        labels = ['0'] * boxes.shape[0]
    else:
        labels = [ class_names[label.item()] for label in labels ]

    print(labels)
    print(type(bb_colors))
    print(type(bb_colors))
    print(type(labels))


    # Denormlaize the image
    img = img.squeeze()
    img = img.detach()
    img = denormalize(img)
    img = img * 255
    img = img.to(torch.uint8)
    

    img = draw_bounding_boxes(img, boxes=boxes, labels=labels, colors=bb_colors, width=3)
    img = F.to_pil_image(img)

    img.show()