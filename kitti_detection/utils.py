from kitti_detection.dataset import DataSample, class_names

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.ops.boxes import box_area

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


    # modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)
