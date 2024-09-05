# LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones
# Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni
# Winter Conference on Applications of Computer Vision (WACV), 2025

import numpy as np

__all__ = ["rand_bbox"]


def rand_bbox(
    h: int,
    w: int,
    lam: float,
    rand_func: callable = np.random.uniform,
) -> tuple[int, int, int, int]:
    """randomly sample bbox, used in cutmix"""
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = w * cut_rat
    cut_h = h * cut_rat

    # uniform
    cx = rand_func(0, w)
    cy = rand_func(0, h)

    bbx1 = int(np.clip(cx - cut_w / 2, 0, w))
    bby1 = int(np.clip(cy - cut_h / 2, 0, h))
    bbx2 = int(np.clip(cx + cut_w / 2, 0, w))
    bby2 = int(np.clip(cy + cut_h / 2, 0, h))

    return bbx1, bby1, bbx2, bby2
