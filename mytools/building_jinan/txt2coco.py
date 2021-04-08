'''
@Author: zfp
@Date: 2019-11-28 19:40:07
@LastEditTime : 2019-12-25 22:33:23
@FilePath: /mmdetection/mytools/building_jinan/txt2coco.py
'''
import os
from utils import json_load, anns_load, read_tif_file, makedirs, write_tif, json_dump
try:
    import gdal
except Exception:
    from osgeo import gdal
import cv2
import numpy as np
import shapely.geometry as shgeo

def visual_temp(temp, ouput_path, boxes, rboxes_points, segmentations):
    if temp.ndim == 2:
        colors = [0, 128, 255]
    else:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        temp = temp[..., 2::-1]
    temp = temp.copy()
    for box, rbox_points, segmentation in zip(boxes, rboxes_points, segmentations):
        box = np.array(box, dtype=np.int32)
        cv2.rectangle(temp, (box[0], box[1]), (box[0] + box[2] - 1, box[1] + box[3] - 1), colors[0], 1)
        rbox_points = np.array(rbox_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(temp, [rbox_points], True, colors[1], 1)
        segmentation = np.array(segmentation, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(temp, [segmentation], True, colors[2], 1)
    cv2.imwrite(ouput_path, temp)
    
    
def txt2coco(cfg):
    output_dir = cfg.output_dir
    crop_size_cfg = cfg.crop_size
    overlap_size_cfg = cfg.overlap_size
    test_overlap = cfg.test_overlap
    thresh = cfg.thresh
    min_size = cfg.min_size
    visual = cfg.visual
    if not cfg.convert_uint8:
        visual = False

    assert crop_size_cfg > 0 and crop_size_cfg > overlap_size_cfg
    assert overlap_size_cfg >= 0

    files = json_load(os.path.join(output_dir, 'filelist.json'))
    process_types = ['pan_process', 'ms_process', 'fusion_process']
    for process_type in process_types:
        print('begin process %s files' % process_type)
        output_dir = os.path.join(cfg.output_dir, process_type.split('_')[0])
        makedirs(output_dir)
        output_path = {}
        result = {}
        for item in ['train', 'val']:
            item_dir = os.path.join(output_dir, item)
            item_image_dir = os.path.join(item_dir, 'JPEGImages')
            item_visual_dir = os.path.join(item_dir, 'visual')
            makedirs(item_dir)
            makedirs(item_image_dir)
            if visual:
                makedirs(item_visual_dir)
            output_path[item] = {}
            output_path[item]['output_dir'] = item_dir
            output_path[item]['image_dir'] = item_image_dir
            output_path[item]['visual_dir'] = item_visual_dir 

            result[item] = {}
            result[item]['images'] = []
            result[item]['annotations'] = []
            result[item]['categories'] = []
            result[item]['categories'].append({'supercategory': 'none', 'id': 1, 'name': 'building'})

        img_id = 0
        ann_id = 0

        for k, v in files.items():
            print(k)
            id_ = v['id']
            width, height = v['size']
            ann_txt = v['txt']
            split = v['split']
            img_file = v[process_type]

            anns = anns_load(ann_txt)
            srcArray = read_tif_file(img_file)
            if not cfg.convert_uint8:
                srcArray = srcArray[[2, 1, 0, 3]]
            assert srcArray is not None
            if srcArray.ndim == 3:
                # C, H, W
                srcArray = srcArray.transpose([1, 2, 0])
            img = cv2.resize(srcArray, (width, height))
            crop_size = crop_size_cfg
            if split == 'train' or test_overlap:
                overlap_size = overlap_size_cfg
            else:
                overlap_size = 0

            for row in range(0, height, crop_size - overlap_size):
                for col in range(0, width, crop_size - overlap_size):
                    if (width - col <= overlap_size or height - row <= overlap_size) and (row > 0 and col > 0):
                        continue
                    start_row = row
                    start_col = col
                    tile_height = min(height - start_row, crop_size)
                    tile_width = min(width - start_col, crop_size)

                    if split == 'train' or test_overlap:
                        if tile_height < crop_size and start_row > 0:
                            tile_height = crop_size
                            start_row = height - crop_size
                        if tile_width < crop_size and start_col > 0:
                            tile_width = crop_size
                            start_col = width - crop_size
                    
                    boxes = []
                    segmentations = []
                    areas = []
                    rboxes = []
                    rboxes_points = []

                    for ann in anns:
                        if len(ann) < 6:
                            continue
                        poly = shgeo.Polygon([(ann[i], ann[i + 1]) for i in range(0, len(ann), 2)])
                        rectangle = shgeo.Polygon(
                            [(start_col, start_row), 
                            (start_col + tile_width - 1, start_row),
                            (start_col + tile_width - 1, start_row + tile_height - 1), 
                            (start_col, start_row + tile_height - 1)])
                        if not (poly.is_valid and rectangle.is_valid):
                            continue

                        inter_polys = poly.intersection(rectangle)
                        if type(inter_polys) != shgeo.multipolygon.MultiPolygon and type(inter_polys) != shgeo.Polygon:
                            continue
                        if type(inter_polys) == shgeo.Polygon:
                            inter_polys = [inter_polys]

                        for inter_poly in inter_polys:
                            if inter_poly.area / poly.area < thresh:
                                continue
                            inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                            segmentation = np.array(list(inter_poly.exterior.coords)[0: -1])

                            for p in segmentation:
                                p[0] = p[0] - start_col
                                p[1] = p[1] - start_row

                            x1 = np.min(segmentation[:, 0])
                            x2 = np.max(segmentation[:, 0])
                            y1 = np.min(segmentation[:, 1])
                            y2 = np.max(segmentation[:, 1])

                            if min(y2 - y1 + 1, x2 - x1 + 1) < min_size:
                                continue

                            areas.append(inter_poly.area)
                            segmentations.append(segmentation.reshape(-1).tolist())
                            boxes.append([x1, y1, x2 - x1 + 1, y2 - y1 + 1])
                            rbox = cv2.minAreaRect(segmentation.astype(np.int).reshape(-1, 2))
                            cx, cy = rbox[0]
                            w, h = rbox[1]
                            angle = rbox[2]
                            rboxes.append([cx, cy, w, h, angle])
                            rboxes_points.append(cv2.boxPoints(rbox))

                    image_item = {}
                    temp = img[start_row: start_row + tile_height, start_col: start_col + tile_width]
                    file_name = '%d_%d_%d' % (id_, start_row, start_col)
                    image_item['file_name'] = file_name + '.tif'
                    image_item['width'] = tile_width
                    image_item['height'] = tile_height
                    image_item['id'] = img_id
                    result[split]['images'].append(image_item)
                    for i in range(len(areas)):
                        annotation_item = {}
                        annotation_item['iscrowd'] = 0
                        annotation_item['category_id'] = 1
                        annotation_item['ignore'] = 0
                        annotation_item['area'] = areas[i]
                        annotation_item['segmentation'] = [segmentations[i]]
                        annotation_item['bbox'] = boxes[i]
                        annotation_item['rbox'] = rboxes[i]
                        annotation_item['image_id'] = img_id
                        annotation_item['id'] = ann_id
                        result[split]['annotations'].append(annotation_item)
                        ann_id += 1
                    img_id += 1
                    if temp.ndim == 3:
                        write_tif(os.path.join(output_path[split]['image_dir'], image_item['file_name']), temp.transpose([2, 0, 1]))
                    else:
                        write_tif(os.path.join(output_path[split]['image_dir'], image_item['file_name']), temp)
                    if visual:
                        visual_temp(temp, os.path.join(output_path[split]['visual_dir'], '%s.png' % file_name), boxes, rboxes_points, segmentations)
        for k, v in result.items():
            json_dump(v, os.path.join(output_path[k]['output_dir'], '%s.json' % k))

if __name__ == '__main__':
    import mytools.building_jinan.cfg as cfg
    txt2coco(cfg)