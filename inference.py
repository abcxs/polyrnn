import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from pycocotools.coco import COCO
site = 'val'
img_dir = f'./data/yunnan_512/{site}/JPEGImages'
json_path = f'./data/yunnan_512/{site}/{site}.json'

coco = COCO(json_path)
img_ids = set(_['image_id'] for _ in coco.anns.values())

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import glob
work_dir = './venus/3'
config_file = glob.glob(os.path.join(work_dir, '*.py'))[0]
checkpoint_file = os.path.join(work_dir, 'latest.pth')
model = init_detector(config_file, checkpoint_file, device='cuda:0')

import numpy as np
import mmcv
import cv2
import torch
def show_result(img_path, result, score_thr):
    img = cv2.imread(img_path)
    bbox_result, segm_result = result
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
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        mask = segms[i].astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask[..., None], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0,0,255), 2)
    return img

import matplotlib.pyplot as plt
from tqdm import tqdm
img_ids = list(img_ids)

for bi, img_id in enumerate(tqdm(img_ids)):
    fn = coco.load_imgs([img_id])[0]['file_name']
    img_path = os.path.join(img_dir, fn)
    result = inference_detector(model, img_path)
#     show_result_pyplot(model, img_path, result, score_thr=0.6)
#     img = show_result(img_path, result, score_thr=0.6)
#     img = img[:, :, ::-1]
#     plt.figure(figsize=(15, 10))
#     plt.axis('off')
#     plt.imshow(img)
#     plt.show()
#     out_path = os.path.join(output_dir, fn)
#     cv2.imwrite(out_path, img)
    if bi > 20:
        break