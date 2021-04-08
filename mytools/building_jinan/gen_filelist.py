import os
import glob
import numpy as np
import json
from utils import walk, geo2cr, json_dump, check_tifs, makedirs
import random
import shutil

try:
    import ogr
    import gdal
except Exception:
    from osgeo import ogr
    from osgeo import gdal
    
ogr.RegisterAll()
gdal.SetConfigOption('SHAPE_ENCODING', "UTF8")
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

def gen_fielist(cfg):
    input_dir = cfg.input_dir
    output_dir = cfg.output_dir
    
    makedirs(output_dir)
    if cfg.filelist is not None:
        shutil.copyfile(cfg.filelist, os.path.join(cfg.output_dir, 'filelist.json'))
        return
    
    seed = cfg.seed
    random.seed(seed)
    percent = cfg.percent

    dir1 = '融合图像'
    dir2 = '高分图像'
    dir3 = '多光谱'

    fusion_dir = os.path.join(input_dir, dir1)
    pan_dir = os.path.join(input_dir, dir2)
    ms_dir = os.path.join(pan_dir, dir3)

    fusion_files = sorted(glob.glob(os.path.join(fusion_dir, '*.tif')))
    tif_filenames = [os.path.splitext(os.path.basename(fusion_file))[0] for fusion_file in fusion_files]

    files = {}
    for tif_filename, fusion_file in zip(tif_filenames, fusion_files):
        files[tif_filename] = {}
        files[tif_filename]['fusion'] = fusion_file

    for tif_filename in tif_filenames:
        tif_file, shp_file = walk(os.path.join(pan_dir, tif_filename))
        files[tif_filename]['pan'] = tif_file
        files[tif_filename]['shp'] = shp_file
        ms_file = os.path.join(ms_dir, '%s.tif' % tif_filename)
        assert os.path.exists(ms_file), '{} doesn\'t exist'.format(ms_file)
        files[tif_filename]['ms'] = ms_file
    # check file exist and open
    files = check_tifs(files)

    ids = list(range(len(files.keys())))
    train_ids = random.sample(ids, int(percent * len(ids)))

    train_images = []
    val_images = []

    for i, (k, v) in enumerate(files.items()):
        v['id'] = i
        if i in train_ids:
            v['split'] = 'train'
            train_images.append(k)
        else:
            v['split'] = 'val'
            val_images.append(k)
    
    
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_images))

    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_images))

    json_dump(files, os.path.join(output_dir, 'filelist.json'))