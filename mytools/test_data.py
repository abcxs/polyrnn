# %%
from pycocotools.coco import COCO
json_file = '/home/zhoufeipeng/code/vertex/data/building/yunnan/train/train.json'
coco = COCO(json_file)
img_ids = coco.get_img_ids()
nums = []
ids = []
for img_id in img_ids:
    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    anns = coco.load_anns(ann_ids)
    for ann in anns:
        num = len(ann['segmentation'][0]) // 2
        nums.append(num)
        if num > 100:
            ids.append(img_id)
            break
        
# %%
sum(nums) / len(nums)
# %%
import matplotlib.pyplot as plt 
plt.hist(nums, 10)
plt.show()
max(nums)
# %%
id_ = ids[0]
info = coco.load_imgs([id_])[0]
info['file_name']
# %%
import os
from PIL import Image
img_dir = '/home/zhoufeipeng/data/building/dst/yunnan/train/JPEGImages'
img_path = os.path.join(img_dir, info['file_name'])
img = Image.open(img_path)
display(img)
# %%
import cv2
import numpy as np
ann_ids = coco.get_ann_ids(img_ids=[id_])
anns = coco.load_anns(ann_ids)
img = cv2.imread(img_path)
for ann in anns:
    poly = np.array(ann['segmentation'][0]).astype(np.int32).reshape(-1, 2)
    if len(poly) > 10:
        print(len(poly))
    # print(len(poly))
        poly = cv2.approxPolyDP(poly, 1, False)[:, 0, :]
        print(poly.min(axis=0), poly.max(axis=0))

        print(len(poly))
    # print(len(poly))
        cv2.polylines(img, [poly], 1, (255, 0, 0), 2)
display(Image.fromarray(img[:, :, ::-1]))
# %%

# %%
