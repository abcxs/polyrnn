import cv2
import os
import json
from collections import defaultdict
filelist_path = '/data/zfp/data/jinan/filelist.json'
img_path = '/data/zfp/data/jinan/visual/%s_fusion.png'
builidng_txt = '/data/zfp/code/mmdetection/results_merge/building/building.txt'
output_dir = '/data/zfp/code/mmdetection/visual/building'
score_thresh = 0.7

result = defaultdict(list)
with open(builidng_txt) as f:
    content = f.read().strip().split('\n')
    content = [c.strip().split() for c in content]
    for c in content:
        id_ = int(c[0])
        score = float(c[1])
        bbox = list(map(int, map(float, c[2:])))
        if score > score_thresh:
            result[id_].append(bbox)

with open(filelist_path) as f:
    filelist = json.load(f)
print(list(filelist.items())[0])
id2name = dict()
valset = []
for k, v in filelist.items():
    id2name[v['id']] = k
    if v['split'] == 'val':
        valset.append(v['id'])

for val_id in valset:
    img = cv2.imread(img_path % id2name[val_id])
    for bbox in result[val_id]:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_dir, id2name[val_id] + '.png'), img)
