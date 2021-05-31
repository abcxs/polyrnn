import multiprocessing
import os
import random
import shutil

import cv2
import numpy as np
import ogr

from utils import json_dump, json_load, get_root_logger

logger = get_root_logger()


def shp2txt(jpgFile, annFile):
    ogr.RegisterAll()

    image = cv2.imread(jpgFile)
    height, width = image.shape[:2]

    dataSource = ogr.Open(annFile)
    daLayer = dataSource.GetLayer(0)
    featureCount = daLayer.GetFeatureCount()

    msgs = []
    for _ in range(featureCount):
        feature = daLayer.GetNextFeature()

        geometry = feature.GetGeometryRef()
        if geometry == None:
            continue

        ring = geometry.GetGeometryRef(0)
        numPoints = ring.GetPointCount()
        if numPoints < 4:
            continue
        msg = []
        points = []
        for j in range(numPoints - 1):
            dcol, drow = ring.GetX(j), abs(ring.GetY(j))

            dcol = max(min(dcol, width - 1), 0)
            drow = max(min(drow, height - 1), 0)
            msg.append(str(dcol))
            msg.append(str(drow))
            points.append([dcol, drow])

        points = np.array(points).reshape(-1, 2)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)

        if msg and max_y - min_y > min_box_size and max_x - min_x > min_box_size:
            msgs.append(' '.join(msg))
    return msgs


def batShp2txt(process_id, files, output_dir, convert_results):
    for key, file in files.items():
        id_ = file['id']
        img_files = file['img']
        ann_files = file['ann']
        backgroud = file['background']

        convert_files = []
        if backgroud:
            for num_id, img_file in enumerate(img_files):
                output_img_path = os.path.join(
                    output_dir, 'tmp', '%d_%d.jpg' % (id_, num_id))
                output_txt_path = os.path.join(
                    output_dir, 'tmp', '%d_%d.txt' % (id_, num_id))
                shutil.copyfile(img_file, output_img_path)
                with open(output_txt_path, 'w') as f:
                    f.write('')
                convert_files.append(
                    {'img_path': output_img_path, 'txt_path': output_txt_path})
        else:
            num_id = 0
            for img_file in img_files:
                for ann_file in ann_files:
                    logger.info(
                        f'process_id: {process_id}, id: {id_}, img_file: {img_file}, ann_file: {ann_file}')
                    msgs = shp2txt(img_file, ann_file)
                    if len(msgs) > 0:
                        output_img_path = os.path.join(
                            output_dir, 'tmp', '%d_%d.jpg' % (id_, num_id))
                        output_txt_path = os.path.join(
                            output_dir, 'tmp', '%d_%d.txt' % (id_, num_id))
                        with open(output_txt_path, 'w') as f:
                            f.write('\n'.join(msgs))
                        shutil.copyfile(img_file, output_img_path)
                        num_id += 1
                        convert_files.append(
                            {'img_path': output_img_path, 'txt_path': output_txt_path})
        convert_results[key] = {'convert': convert_files}


def multiShp2txt(cfg):
    output_dir = cfg.output_dir
    num_process = cfg.num_process
    global min_box_size
    min_box_size = cfg.min_box_size

    files = json_load(os.path.join(output_dir, 'filelist.json'))
    keys = list(files.keys())

    manager = multiprocessing.Manager()
    convert_results = manager.dict()
    processes = []

    for i in range(num_process):
        process_files = {}
        for key in keys[i::num_process]:
            process_files[key] = files[key]
        p = multiprocessing.Process(target=batShp2txt, args=(
            i, process_files, output_dir, convert_results))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    for k, v in convert_results.items():
        files[k].update(v)
    json_dump(files, os.path.join(output_dir, 'filelist.json'))


if __name__ == '__main__':
    pass
