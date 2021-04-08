# %%
import os
import glob
import ogr
import gdal
import sys
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt
from mytools.building_jinan.utils import walk, geo2cr, check_tifs
ogr.RegisterAll()
gdal.SetConfigOption('SHAPE_ENCODING', "UTF8")
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

min_box_size = 4

input_dir = '/data/zfp/data/济南大学'
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
    assert os.path.exists(ms_file), '{} doesn\'t exist'
    files[tif_filename]['ms'] = ms_file

files = check_tifs(files)

for k, v in files.items():
    print(k)
    print(v)
    break

# %%
cate2 = {'住房': 'Hb', '厂房': 'Fb', '大棚': 'Gh', '其他1': 'OI', '其他2': 'OII', None: 'OII'}
file_polys = {}
files_fail = []
polys_fail = defaultdict(int)
num_per_cate = defaultdict(int)
for k, v in files.items():
    print(k)
    file_polys[k] = {}
    file_polys[k]['points'] = []
    file_polys[k]['box'] = []
    file_polys[k]['rbox'] = []

    pan_tif_file = v['pan']
    ms_tif_file = v['ms']
    scale_w, scale_h = 1, 1
    # scale_w, scale_h = get_scale(pan_tif_file, ms_tif_file)

    pan_tif_file = v['pan']
    dataset = gdal.Open(pan_tif_file)
    if dataset is None:
        print('fail to open {}'.format(pan_tif_file))
        files_fail.append(k)
        continue

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geoTransform = dataset.GetGeoTransform()
    if geoTransform is None:
        print('{}: geoTransfrom is None'.format(pan_tif_file))
        files_fail.append(k)
        continue

    shp_file = v['shp']
    dataSource = ogr.Open(shp_file)
    if dataSource is None:
        print('fail to open {}'.format(shp_file))
        files_fail.append(k)
        continue
    daLayer = dataSource.GetLayer(0)
    featureCount = daLayer.GetFeatureCount()
    
    daLayer.ResetReading()
    for _ in range(featureCount):
        feature = daLayer.GetNextFeature()
        fieldName = feature.GetField('类型')
        if fieldName == '其它2':
            fieldName = '其他2'
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
        
        points = np.array(points).reshape(-1, 2) * [scale_w, scale_h]
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        box = [x_min, y_min, w, h]
        
        rect = cv2.minAreaRect(points.astype(np.int))
        x_c, y_c = rect[0]
        w, h = rect[1]
        a = rect[2]
        rbox = [x_c, y_c, w, h, a]

        if min(box[2], box[3]) < min_box_size:
            print('{}: the poly is so small, FID:{}, width:{}, height:{}'.format(shp_file, feature.GetFID(), box[2], box[3]))
            polys_fail[k] += 1
            continue
        num_per_cate[cate2[fieldName]] += 1
        file_polys[k]['box'].append(box)
        file_polys[k]['points'].append(points)
        file_polys[k]['rbox'].append(rbox)
# %%
print(num_per_cate)
x=np.arange(len(num_per_cate.keys()))
plt.bar(x, list(num_per_cate.values()))
ax=plt.gca()
labels = list(num_per_cate.keys())
labels.insert(0, '1')

ax.set_xticklabels(labels)
plt.savefig(os.path.abspath('cate.png'))
plt.show()
print('error with the num of files:', len(files_fail))
print('error with the num of polys:', sum(polys_fail.values()))
num_building = 0
for k, v in file_polys.items():
    num_building += len(file_polys[k]['points'])
print('the num of polys:', num_building)
# error with the num of files: 0
# error with the num of polys: 72
# the num of polys: 62177
# %%
# the angle of instance
angles = []
for k, v in file_polys.items():
    for rbox in v['rbox']:
        w, h, angle  = rbox[-3:]
        if w < h:
            angle = angle + 90
        angles.append(angle)
plt.hist(angles, rwidth=0.9)
plt.savefig(os.path.abspath('angle.png'))
plt.show()


# %%
import bisect
sizes_range = [0, 32, 96, 1000]
sizes_num = [0 for _ in range(len(sizes_range))]
sizes = []
for k, v in file_polys.items():
    sizes_per_image = [sqrt(box[2] * box[3]) for box in v['box']]
    pos_per_image = [bisect.bisect_right(sizes_range, size) for size in sizes_per_image]
    for pos in pos_per_image:
        sizes_num[pos - 1] += 1
    sizes.extend(sizes_per_image)
print(sizes_num)
plt.hist(sizes, bins=10, rwidth=0.8)
plt.savefig(os.path.abspath('instance_size.png'))
plt.show()

# %%
img_sizes = []
for k, v in files.items():
    pan_file = v['pan']
    dataset = gdal.Open(pan_file)
    img_sizes.append(sqrt(dataset.RasterXSize * dataset.RasterYSize))
plt.hist(img_sizes, bins=10, rwidth=0.8)
plt.savefig(os.path.abspath('img_size.png'))
plt.show()
# %%
bbox_ratios = []
rbox_ratios = []
for k, v in file_polys.items():
    for box in v['box']:
        w = box[2]
        h = box[3]
        big = max(w, h)
        small = min(w, h)
        if big / small > 15:
            continue
        bbox_ratios.append(big / small)
        
    for rbox in v['rbox']:
        w = rbox[2]
        h = rbox[3]
        big = max(w, h)
        small = min(w, h)
        if big / small > 15:
            continue
        rbox_ratios.append(big / small)
plt.hist(bbox_ratios, bins=15, rwidth=0.8)
plt.savefig(os.path.abspath('bbox_ratio.png'))
plt.show()
plt.hist(rbox_ratios, bins=15, rwidth=0.8)
plt.savefig(os.path.abspath('rbox_ratio.png'))
plt.show()

# %%
