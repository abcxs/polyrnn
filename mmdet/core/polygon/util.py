
import numpy as np
import cv2
import torch

def get_edge_mask(poly, mask):
    cv2.polylines(mask, [poly], True, [1])
    return mask

def class_to_grid(poly, out_tensor):

    """
    用来转换第一个点
    poly: [batch, ]
    out_tensor: [batch, 1, grid_size, grid_size]
    """
    out_tensor.zero_()
    h, w = out_tensor.shape[-2:]

    for i, p in enumerate(poly):
        if p < h * w:
            x = (p % w).long()
            y = (p // w).long()
            out_tensor[i, 0, y, x] = 1
    return out_tensor

def poly01_to_poly0g(poly, grid_size, epsilon=1, remove_abundant=False):
    """
    [0, 1] coordinates to [0, grid_size] coordinates

    Note: simplification is done at a reduced scale
    """
    # 不一定好，通过floor，偏移距离问题
    poly_ = poly * grid_size
    poly = np.floor(poly_).astype(np.int32)
    offset = poly_ - poly
    # 距离值 暂时定为1
    # 值越大，需要拟合的点越少，精度会下降
    poly = cv2.approxPolyDP(poly, epsilon, False)[:, 0, :]
    if not remove_abundant:
        return poly, offset
    valid = []
    for i in range(len(poly)):
        last = (i + 1) % len(poly)
        if not point_in_line(poly[i - 1], poly[i], poly[last]):
            valid.append(i)
    valid = np.array(valid, dtype=np.long)
    poly_ = poly_[valid]
    poly = np.floor(poly_).astype(np.int32)
    offset = poly_ - poly
    return poly, offset

def point_in_line(p1, p2, p3):
    v1 = p1[0] * p2[1] - p2[1] * p1[1]
    v2 = p2[0] * p3[1] - p3[0] * p2[1]
    v3 = p3[0] * p1[1] - p3[1] * p1[0]
    v = v1 + v2 + v3
    return v == 0

def xy_to_class(poly, grid_size):
    """
    NOTE: Torch function
    poly: [n, time_steps, 2]
    
    Returns: [time_steps] with class label
    for x,y location or EOS token
    """
    w, h = grid_size

    poly[..., 1] *= w
    poly = torch.sum(poly, dim=-1)

    poly[poly < 0] = w * h
    return poly