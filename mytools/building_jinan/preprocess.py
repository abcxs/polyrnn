# %%
import os
from collections import defaultdict
import mytools.building_jinan.utils as utils
from .utils import geo2cr, json_load, json_dump, makedirs, read_tif_file, cv2_save, write_tif
import numpy as np
import math

try:
    import gdal
    import ogr
except Exception:
    from osgeo import gdal
    from osgeo import ogr
ogr.RegisterAll()
gdal.SetConfigOption('SHAPE_ENCODING', "UTF8")
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

def shp2txt(cfg):
    min_size = cfg.min_size
    scale = cfg.scale
    output_dir = cfg.output_dir

    makedirs(os.path.join(output_dir, 'txt'))
    files = json_load(os.path.join(output_dir, 'filelist.json'))

    files_fail = []
    polys_fail = defaultdict(int)
    building_num = 0
    for k, v in files.items():
        # Panchromatic match annotated file
        pan_file = v['pan']
        shp_file = v['shp']

        dataset = gdal.Open(pan_file)
        assert dataset is not None
        
        width = dataset.RasterXSize
        height = dataset.RasterYSize

        v['size'] = [width, height]

        geoTransform = dataset.GetGeoTransform()
        if geoTransform is None:
            print('{}: geoTransfrom is None'.format(pan_file))
            files_fail.append(k)
            continue

        dataSource = ogr.Open(shp_file)
        if dataSource is None:
            print('fail to open {}'.format(shp_file))
            files_fail.append(k)
            continue
        daLayer = dataSource.GetLayer(0)
        featureCount = daLayer.GetFeatureCount()

        points_per_file = []

        daLayer.ResetReading()
        for _ in range(featureCount):
            feature = daLayer.GetNextFeature()
            geometry = feature.GetGeometryRef()
            if geometry is None:
                print('{}: geometry is None'.format(shp_file))
                polys_fail[k] += 1
                continue
            geometryType = geometry.GetGeometryType()
            if geometryType != ogr.wkbPolygon:
                print('{}: the FID is {}, the type of geometry is {}'.format(shp_file, feature.GetFID(), geometryType))
                polys_fail[k] += 1
                continue
            geometryCount = geometry.GetGeometryCount()
            if geometryCount != 1:
                print('{}: poly has {} rings'.format(shp_file, geometryCount))
                polys_fail[k] += 1

            ring = geometry.GetGeometryRef(0)
            numPoints = ring.GetPointCount()
            if numPoints < 4:
                print('{} : the num of ring is less than 3'.format(shp_file))
                polys_fail[k] += 1
                continue

            points = []
            max_y = 0
            max_x = 0
            min_y = height - 1
            min_x = width - 1
            for i in range(numPoints - 1):
                x, y = geo2cr(geoTransform, ring.GetX(i), ring.GetY(i))
            
                x = max(min(x, width - 1), 0)
                y = max(min(y, height - 1), 0)

                points.extend([x, y])
        
            points_array = np.array(points).reshape(-1, 2) * scale
            x_min, y_min = points_array.min(axis=0)
            x_max, y_max = points_array.max(axis=0)
            w = x_max - x_min + 1
            h = y_max - y_min + 1
        
            if min(w, h) < min_size:
                print('{}: the poly is so small, width:{}, height:{}'.format(shp_file, w, h))
                polys_fail[k] += 1
                continue

            points = [str(p) for p in points]
            points_per_file.append(' '.join(points))
            building_num += 1

        with open(os.path.join(output_dir, 'txt', '%s.txt' % k), 'w') as f:
            f.write('\n'.join(points_per_file)) 

        v['txt'] = os.path.join(output_dir, 'txt', '%s.txt' % k)
    
    json_dump(files, os.path.join(output_dir, 'filelist.json'))
    print('error with the num of files:', len(files_fail))
    print('error with the num of buildings:', sum(polys_fail.values()))
    print('the num of buildings:', building_num)

def tif2uint8(cfg):
    output_dir = cfg.output_dir
    visual = cfg.visual
    transform = cfg.transform
    tmp_dir = os.path.join(output_dir, 'tmp')
    makedirs(tmp_dir)
    visual_dir = os.path.join(output_dir, 'visual')
    if visual:
        makedirs(visual_dir)
    transform = getattr(utils, transform)
    files = json_load(os.path.join(output_dir, 'filelist.json'))
    items = ['ms', 'pan', 'fusion']
    for k, v in files.items():
        print(k)
        for item in items:
            print(item)
            tif_file = v[item]
            tif_filename = os.path.splitext(os.path.basename(tif_file))[0]
            img = read_tif_file(tif_file)
            assert img is not None
            img = transform(img)
            if img.ndim == 3:
                # convert to RGBI
                img = img[[2, 1, 0, 3]]
            output_tmp_file = os.path.join(tmp_dir, '%s_%s.tif' % (tif_filename, item))
            write_tif(output_tmp_file, img)
            v['%s_process' % item] = output_tmp_file
            if visual:
                output_visual_file = os.path.join(visual_dir, '%s_%s.png' % (tif_filename, item))
                cv2_save(output_visual_file, img)
                v['%s_visual' % item] = output_visual_file
    json_dump(files, os.path.join(output_dir, 'filelist.json'))

def tif2uint16(cfg):
    output_dir = cfg.output_dir
    files = json_load(os.path.join(output_dir, 'filelist.json'))
    items = ['ms', 'pan', 'fusion']
    for k, v in files.items():
        print(k)
        for item in items:
            print(item)
            tif_file = v[item]
            v['%s_process' % item] = tif_file
    json_dump(files, os.path.join(output_dir, 'filelist.json'))


def preprocess(cfg):
    if cfg.convert_uint8:
        tif2uint8(cfg)
    else:
        tif2uint16(cfg)
    shp2txt(cfg)
