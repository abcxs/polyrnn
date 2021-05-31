import os
import cv2
import numpy as np
import shutil
import multiprocessing
from utils import json_load, get_root_logger

logger = get_root_logger()

def visual_tmp(process_id, files):
    for k, v in files.items():
        id_ = v['id']
        logger.info(f'process: {process_id}, id: {id_}, path: {k}')
        convert_results = v['convert']
        for convert_result in convert_results:
            img_path = convert_result['img_path']
            txt_path = convert_result['txt_path']
            img = cv2.imread(img_path)
            anns = open(txt_path).read().strip().split('\n')
            anns = [[float(p) for p in ann.strip().split(' ')] for ann in anns if ann]
            for ann in anns:
                ann = np.array(ann, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [ann], True, (0, 0, 255), 1)
            cv2.imwrite(img_path.replace('tmp', 'tmp_visual', 1), img)

def visual(cfg):
    output_dir = cfg.output_dir
    num_process = cfg.num_process
    files = json_load(os.path.join(output_dir, 'filelist.json'))
    output_dir = os.path.join(output_dir, 'tmp_visual')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    processes = []
    keys = list(files.keys())
    for i in range(num_process):
        process_files = {}
        for key in keys[i::num_process]:
            process_files[key] = files[key]
        p = multiprocessing.Process(target=visual_tmp, args=(i, process_files))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    

if __name__ == '__main__':
    import cfg
    visual(cfg)