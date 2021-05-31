import os
import random

from utils import json_dump, get_root_logger

logger = get_root_logger()


def walk(cfg):
    input_dir = cfg.input_dir
    output_dir = cfg.output_dir
    seed = cfg.seed
    percent = cfg.percent

    random.seed(seed)
    files = {}

    l1s = os.listdir(input_dir)
    for l1 in l1s:
        if os.path.isdir(os.path.join(input_dir, l1)):
            result = walk_and_check(os.path.join(input_dir, l1))
            if result is None:
                continue
            if 'background' in l1:
                result['ann'] = []
                result['background'] = True
            else:
                result['background'] = False
            files[os.path.join(input_dir, l1)] = result

    # 进行排序，防止遍历结果的影响
    files = dict(sorted(files.items()))
    for i, f in enumerate(files.values()):
        f['id'] = i
    ids = [data['id'] for data in files.values() if not data['background']]
    val_ids = set(random.sample(ids, int(len(ids) * (1 - percent))))

    for data in files.values():
        if data['id'] in val_ids:
            data['split'] = 'val'
        else:
            data['split'] = 'train'
    json_dump(files, os.path.join(output_dir, 'filelist.json'))


def walk_and_check(path):
    result_file = {}
    result_file['img'] = []
    result_file['ann'] = []

    for root, _, files in os.walk(path):
        for item in files:
            if item.endswith('.jpg'):
                result_file['img'].append(os.path.join(root, item))
            elif item.endswith('.shp'):
                result_file['ann'].append(os.path.join(root, item))
    if len(result_file['img']) > 0 and len(result_file['ann']) > 0:
        info = 'len(img):%d, len(ann):%d' % (len(result_file['img']), len(result_file['ann']))
        logger.info(info)
        return result_file
    else:
        info = 'len(img):%d, len(ann):%d' % (len(result_file['img']), len(result_file['ann']))
        info = f'skip {root}, it may have zero shps or zero tifs, ' + info
        logger.warning(info)

if __name__ == '__main__':
    import cfg
    walk(cfg)
