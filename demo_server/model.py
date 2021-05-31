import os
from mmdet.apis import inference_detector, init_detector
import numpy as np
import cv2
import mmcv
import torch
import tifffile as tiff

class Model(object):
    def __init__(self, config_file, checkpoint_file, score_thr, gpu_id):
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.score_thr = score_thr
        self.gpu_id = gpu_id
        self.model = init_detector(config_file, checkpoint_file, device='cpu')
        self.model = self.model.to(f'cuda:{gpu_id}')

    def show_result(self, img_path, result, ret_det=False):
        return show_result(img_path, result, self.score_thr, ret_det=ret_det)
    
    def __call__(self, img_path):
        result = inference_detector(self.model, img_path)
        return result

def show_mask_result(img, result, score_thr):
    # 基于掩码，提取轮廓显示
    bbox_result, segm_result = result[:2]
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    if segms is not None:
        segms = segms[inds, ...]
    points = []
    for i in range(len(bboxes)):
        mask = segms[i].astype(np.uint8)
        contours, _ = cv2.findContours(mask[..., None], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0,0,255), 2)
        for contour in contours:
            points.append(len(contour))
    return img, sum(points) / len(points)

def show_polygon_result(img, result, score_thr):
    # 根据点结果，显示轮廓
    bbox_result, _, polygon_result = result

    bboxes = np.vstack(bbox_result)
    scores = bboxes[:, -1]
    inds = np.nonzero(scores > score_thr)[0]
    points = []
    for i in inds:
        poly = np.array(polygon_result[0][i]).reshape(-1, 2).astype(np.int32)
        points.append(len(poly))
        cv2.polylines(img, [poly], True, (0, 0, 255), 2)
    return img, sum(points) / len(points)

def show_det_result(img, result, score_thr):
    bbox_result = result
    bboxes = np.vstack(bbox_result)
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    for box in bboxes:
        x1, y1, x2, y2 = box.astype(np.int32).tolist()[:4]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img

def show_result(img_path, result, score_thr, ret_det=False):
    if os.path.splitext(img_path)[1].lower() == '.tif':
        img = tiff.imread(img_path)
        img = img[:, :, :3][:, :, ::-1]
        img = img.copy()
    else:
        img = cv2.imread(img_path)
    if ret_det:
        return show_det_result(img, result, score_thr=score_thr)
    if len(result) == 3:
        return show_polygon_result(img, result, score_thr=score_thr)
    else:
        return show_mask_result(img, result, score_thr=score_thr)