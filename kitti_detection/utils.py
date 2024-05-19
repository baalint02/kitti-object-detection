from kitti_detection.dataset import DataSample, class_names

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

def display_samples_v(samples):
    plt.rcParams["savefig.bbox"] = 'tight'

    bb_colors = ['blue', 'yellow', 'green', 'red', 'orange'] * 8
    
    if not isinstance(samples, tuple):
        samples = (samples)

    _, axs = plt.subplots(nrows=len(samples), squeeze=False, figsize=(12, 3.3 * len(samples)))

    plt.tight_layout()
    for i, sample in enumerate(samples):
        img, target = sample
        labels = [ class_names[label.item()] for label in target['labels'] ]

        img = img.detach()
        img = draw_bounding_boxes(img, boxes=target['boxes'], labels=labels, colors=bb_colors, width=3)
        img = F.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.show()

def display_samples_h(samples):
    plt.rcParams["savefig.bbox"] = 'tight'

    bb_colors = ['blue', 'yellow', 'green', 'red', 'orange'] * 8
    
    if not isinstance(samples, tuple):
        samples = (samples)

    _, axs = plt.subplots(ncols=len(samples), squeeze=False, figsize=(3.3 * len(samples), 12))

    plt.tight_layout()
    for i, sample in enumerate(samples):
        img, target = sample
        labels = [ class_names[label.item()] for label in target['labels'] ]

        img = img.detach()
        img = draw_bounding_boxes(img, boxes=target['boxes'], labels=labels, colors=bb_colors, width=3)
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.show()
