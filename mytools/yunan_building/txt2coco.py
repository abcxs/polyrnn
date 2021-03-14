import json
import multiprocessing
import os
import pickle

import cv2
import numpy as np
import shapely.geometry as shgeo
from utils import json_load, get_root_logger, json_dump

logger = get_root_logger()


def divide(process_id, files, cfg):
    output_dir = cfg.output_dir
    thresh = cfg.thresh
    size = cfg.size
    overlap_size_cfg = cfg.overlap_size
    prefix = cfg.prefix
    min_box_size = cfg.min_box_size
    padding = cfg.padding
    assert size > overlap_size_cfg

    result = {}
    result['train'] = []
    result['val'] = []

    for k, v in files.items():

        id_ = v['id']
        logger.info(f'process: {process_id}, id: {id_}, path: {k}')

        split = v['split']
        convert_results = v['convert']

        img_output_dir = os.path.join(output_dir, split, 'JPEGImages')
        visual_output_dir = os.path.join(output_dir, split, 'visual')
        if split == 'train':
            overlap_size = overlap_size_cfg
        else:
            overlap_size = 0

        for convert_result in convert_results:
            img_path = convert_result['img_path']
            txt_path = convert_result['txt_path']
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            img = cv2.imread(img_path)
            height, width, _ = img.shape

            anns = open(txt_path).read().strip().split('\n')
            anns = [[float(p) for p in ann.strip().split(' ')]
                    for ann in anns if ann]

            for row in range(0, height, size - overlap_size):
                for col in range(0, width, size - overlap_size):
                    if width - col <= overlap_size and col != 0:
                        continue
                    if height - row <= overlap_size and row != 0:
                        continue
                    start_row = row
                    start_col = col
                    tile_height = min(height - start_row, size)
                    tile_width = min(width - start_col, size)

                    if split == 'train':
                        if tile_height < size and start_row > 0:
                            tile_height = size
                            start_row = height - size
                        if tile_width < size and start_col > 0:
                            tile_width = size
                            start_col = width - size
                    assert start_row >= 0 and start_col >= 0, f'start_row: {start_row}, start_col: {start_col}'

                    boxes = []
                    segmentations = []
                    areas = []

                    for ann in anns:
                        if len(ann) < 6:
                            continue
                        poly = shgeo.Polygon(
                            [(ann[i], ann[i + 1]) for i in range(0, len(ann), 2)])
                        rectangle = shgeo.Polygon(
                            [(start_col, start_row),
                             (start_col + tile_width - 1, start_row),
                             (start_col + tile_width - 1,
                              start_row + tile_height - 1),
                             (start_col, start_row + tile_height - 1)])
                        if not (poly.is_valid and rectangle.is_valid):
                            continue

                        inter_polys = poly.intersection(rectangle)
                        if type(inter_polys) != shgeo.multipolygon.MultiPolygon and type(inter_polys) != shgeo.polygon.Polygon:
                            continue
                        if type(inter_polys) == shgeo.polygon.Polygon:
                            inter_polys = [inter_polys]

                        for inter_poly in inter_polys:
                            if inter_poly.area / poly.area < thresh:
                                continue
                            inter_poly = shgeo.polygon.orient(
                                inter_poly, sign=1)
                            segmentation = np.array(
                                list(inter_poly.exterior.coords)[:-1])

                            for p in segmentation:
                                p[0] = p[0] - start_col
                                p[1] = p[1] - start_row

                            x1 = np.min(segmentation[:, 0])
                            x2 = np.max(segmentation[:, 0])
                            y1 = np.min(segmentation[:, 1])
                            y2 = np.max(segmentation[:, 1])

                            if y2 - y1 < min_box_size or x2 - x1 < min_box_size:
                                continue

                            areas.append(inter_poly.area)
                            segmentations.append(
                                segmentation.reshape(-1).tolist())
                            boxes.append([x1, y1, x2 - x1 + 1, y2 - y1 + 1])

                    if padding:
                        temp = np.zeros([size, size, 3], dtype=np.uint8)
                        tile_img = img[start_row: start_row +
                                       tile_height, start_col: start_col + tile_width]
                        temp[:tile_img.shape[0], :tile_img.shape[1]] = tile_img
                    else:
                        temp = img[start_row: start_row + tile_height,
                                   start_col: start_col + tile_width]

                    if (temp == 0).all() or (temp == 255).all():
                        continue

                    data_item = {}
                    image = {}
                    file_name = '%s_%s_%d_%d' % (
                        prefix, img_name, start_row, start_col)
                    image['file_name'] = file_name + '.png'
                    image['width'] = temp.shape[1]
                    image['height'] = temp.shape[0]

                    data_item['image'] = image
                    data_item['annotations'] = []

                    for i in range(len(areas)):
                        annotation = {}
                        annotation['iscrowd'] = 0
                        annotation['category_id'] = 1
                        annotation['ignore'] = 0
                        annotation['area'] = areas[i]
                        annotation['difficult'] = 0
                        annotation['segmentation'] = [segmentations[i]]
                        annotation['bbox'] = boxes[i]
                        data_item['annotations'].append(annotation)

                    result[split].append(data_item)
                    cv2.imwrite(os.path.join(img_output_dir,
                                             '%s.png' % file_name), temp)

                    temp = temp.copy()
                    for box in boxes:
                        box = np.array(box, dtype=np.int32)
                        cv2.rectangle(
                            temp, (box[0], box[1]), (box[2] + box[0] - 1, box[3] + box[1] - 1), (255, 0, 0), 1)
                    for segmentation in segmentations:
                        segmentation = np.array(segmentation, dtype=np.int32)
                        cv2.polylines(
                            temp, [segmentation.reshape(-1, 1, 2)], True, (0, 0, 255), 1)
                    cv2.imwrite(os.path.join(visual_output_dir,
                                             '%s.png' % file_name), temp)

    for split in ['train', 'val']:
        with open(os.path.join(output_dir, split, '%d.pkl' % process_id), 'wb') as f:
            pickle.dump(result[split], f)


def multi_divide(cfg):
    output_dir = cfg.output_dir
    num_process = cfg.num_process
    files = json_load(os.path.join(output_dir, 'filelist.json'))

    keys = list(files.keys())

    process = []

    for i in range(num_process):
        process_files = {}
        for key in keys[i::num_process]:
            process_files[key] = files[key]
        p = multiprocessing.Process(
            target=divide, args=(i, process_files, cfg))
        p.start()
        process.append(p)
    for p in process:
        p.join()

    for split in ['train', 'val']:

        result = {}
        result['images'] = []
        result['annotations'] = []
        result['categories'] = []

        result['categories'].append(
            {'supercategory': 'none', 'id': 1, 'name': 'building'})

        img_id = 0
        ann_id = 0

        for i in range(num_process):
            with open(os.path.join(output_dir, split, '%d.pkl' % i), 'rb') as f:
                datas = pickle.load(f)
            for data in datas:
                data['image']['id'] = img_id
                result['images'].append(data['image'])
                for ann in data['annotations']:
                    ann['image_id'] = img_id
                    ann['id'] = ann_id
                    result['annotations'].append(ann)
                    ann_id += 1
                img_id += 1
            os.remove(os.path.join(output_dir, split, '%d.pkl' % i))
        json_dump(result, os.path.join(output_dir, split, f'{split}.json'))


if __name__ == '__main__':
    import cfg
    multi_divide(cfg)
