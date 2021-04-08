# %%
import os
import json
def convert(input_dir):
    result_file = os.path.join(input_dir, 'result.pkl.bbox.json')
    assert os.path.exists(result_file)
    with open(result_file) as f:
        result = json.load(f)
    return result
result = convert('/data/zfp/mmdetection/experiments/result/faster_rcnn_r50_fpn_1x_0100_1')

# %%
import numpy as np
def second_convert(input_dir):
    result_file = os.path.join(input_dir, 'result.pkl.bbox.json')
    assert os.path.exists(result_file)
    with open('/data/zfp/data/convert.json') as f:
        converts = json.load(f)
    with open(result_file) as f:
        result = json.load(f)

    contents = {}
    for k, v in converts.items():
        print(k)
        contents[k] = []
        for item in v:
            for bbox in result:
                if item[0] == bbox['image_id']:
                    row = item[1]
                    col = item[2]
                    # x1, y1, w, h
                    convert_bbox = np.array(bbox['bbox']) + [col, row, 0, 0]
                    convert_bbox = convert_bbox.tolist()
                    contents[k].append([convert_bbox, bbox['score']])
    with open('/data/zfp/data/convert_result.json', 'w') as f:
            json.dump(contents, f, ensure_ascii=False)
second_convert('/data/zfp/mmdetection/experiments/result/faster_rcnn_r50_fpn_1x_0100_1')
# %%
def first_convert():
    filelist_file = '/data/zfp/data/jinan_2/filelist.json'
    annotation_file = '/data/zfp/data/jinan_2/pan/val/val.json'
    with open(filelist_file) as f:
        filelist = json.load(f)
    with open(annotation_file) as f:
        annotations = json.load(f)
    annotations = annotations['images']
    contents = {}
    for k, v in filelist.items():
        if v['split'] == 'val':
            print(k, v['id'])
            contents[k] = []
            for annotation in annotations:
                # 图片标号 行 列
                id_, row, col = [int(_) for _ in os.path.splitext(annotation['file_name'])[0].split('_')]
                img_id = annotation['id']
                if id_ == v['id']:
                    contents[k].append([img_id, row, col])
    with open('/data/zfp/data/convert.json', 'w') as f:
            json.dump(contents, f, ensure_ascii=False)
first_convert()
# %%
import cv2
import os
import json
with open('/data/zfp/data/convert_result.json') as f:
    converts = json.load(f)
img_path = '/data/zfp/data/jinan_2/visual/%s_fusion.png'

for k, v in converts.items():
    img = cv2.imread(img_path % k)
    # img = img[:, :, ::-1]
    img = img.copy()
    print(img.shape)
    print(len(v))
    for bbox in v:
        box = bbox[0]
        box = [int(b) for b in box]
        x, y, w, h = box
        score = bbox[1]
        if score > 0.8:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imwrite('/data/zfp/tt.png', img)
    cv2.imwrite('/data/zfp/visual/%s.png' % k, img)
# %%
print(1)
# %%
