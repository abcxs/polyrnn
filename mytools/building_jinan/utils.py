'''
@Author: zfp
@Date: 2019-11-14 21:11:03
@LastEditTime : 2019-12-24 21:44:44
@FilePath: /mmdetection/mytools/utils.py
'''
import os
try:
    import gdal
except Exception:
    from osgeo import gdal
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
mod = sys.modules[__name__]

gdal.SetConfigOption('SHAPE_ENCODING', "UTF8")
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

def geo2cr(geoTransform, px, py):
    dTemp = geoTransform[1] * geoTransform[5] - geoTransform[2] * geoTransform[4]
    col = (geoTransform[5] * (px - geoTransform[0]) - geoTransform[2] * (py - geoTransform[3])) / dTemp + 0.5
    row = (geoTransform[1] * (py - geoTransform[3]) - geoTransform[4] * (px - geoTransform[0])) / dTemp + 0.5
    return col, row

def cr2geo(geoTransform, col, row):
    px = geoTransform[0] + col * geoTransform[1] + row * geoTransform[2]
    py = geoTransform[3] + col * geoTransform[4] + row * geoTransform[5]
    return px, py

def walk(input_dir):
    tif_files = []
    shp_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('tif'):
                tif_files.append(os.path.join(root, file))
            elif file.endswith('.shp'):
                shp_files.append(os.path.join(root, file))
    assert len(tif_files) == 1 and len(shp_files) == 1, '{} should have 1 tif and 1 shp , but get {} tifs and {} shps'.format(input_dir, len(tif_files), len(shp_files))
    return tif_files[0], shp_files[0]

def json_dump(files, output_file):
    with open(output_file, 'w') as f:
        json.dump(files, f, ensure_ascii=False)

def json_load(input_file):
    with open(input_file, 'r') as f:
        files = json.load(f)
    return files

def anns_load(anns_file, anns_format='txt'):
    assert anns_format == 'txt'
    if anns_format == 'txt':
        anns = open(anns_file, 'r').read().strip().split('\n')
        anns = [ann.strip().split(' ') for ann in anns]
        anns = [[float(pos.strip()) for pos in ann] for ann in anns]
    return anns

def read_tif_file(tif_path):
    dataset = gdal.Open(tif_path)
    srcArray = dataset.ReadAsArray()
    if srcArray is None:
        print('read {} fail'.format(tif_path))
        return None
    return srcArray

def check_tifs(files, items=['pan', 'ms', 'fusion']):
    # check file available
    files_new = {}
    for k, v in files.items():
        for item in items:
            tif_file = v[item]
            if not os.path.exists(tif_file):
                print('%s: %s doesn\'t exist' % (k, item))
                break
            dataset = gdal.Open(tif_file)
            if dataset is None:
                print('%s: %s open fail' % (k, item))
                break
        files_new[k] = v
    return files_new

def write_tif(file_path, img):
    # the dataset of JiNan
    # Blue Green Red near-infrared
    # gray
    if img.ndim == 2:
        img = img[None]
    # C, H, W
    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    bands, height, width = img.shape
    
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(file_path, width, height, bands, datatype)
    assert dataset is not None
   
    for i in range(bands):
        dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dataset
        
def makedirs(output_dir, delete=True):
    if os.path.exists(output_dir):
        if delete:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)

def linear(img, thresh=0.05, pixle_min_vlaue=0, pixel_max_value=5000):
    img = img.copy()
    img[img < pixle_min_vlaue] = pixle_min_vlaue
    img[img > pixel_max_value] = pixel_max_value
    if img.ndim == 2:
        img = img[None]
    for c in range(0, img.shape[0]):
        pixel_num = np.zeros(pixel_max_value + 1)
        for i in range(0, img.shape[1]):
            for j in range(0, img.shape[2]):
                pixel_num[img[c, i, j]] += 1
        prob = np.cumsum(pixel_num / (img.shape[1] * img.shape[2]))
        min_value = np.argmax(prob >= thresh)
        max_value = np.argmax(prob >= 1 - thresh)
        max_id = img[c] > max_value
        min_id = img[c] < min_value
        img[c, max_id] = 255
        img[c, min_id] = 0
        img[c, ~(max_id | min_id)] = (img[c, ~(max_id | min_id)] - min_value) / (max_value - min_value) * 255
    img = img.astype(np.uint8)
    img = img.squeeze()
    return img

def scale(img, pixle_min_vlaue=0, pixel_max_value=5000):
    # may be affect extreme point
    img = img.copy()
    img[img < pixle_min_vlaue] = pixle_min_vlaue
    img[img > pixel_max_value] = pixel_max_value
    if img.ndim == 2:
        img = img[None]
    for c in range(0, img.shape[0]):
        max_value = img[c].max()
        min_value = img[c].min()
        img[c] = (img[c] - min_value) / (max_value - min_value) * 255
    img = img.astype(np.uint8)
    img = img.squeeze()
    return img

def cv2_save(file_path, img):
    # used for test show
    # C, H, W
    # expect the format R B G I
    if img.ndim == 3:
        img = img.transpose([1, 2, 0])
        cv2.imwrite(file_path, img[:, :, 2::-1])
    else:
        cv2.imwrite(file_path, img)


def gdal_show(tif_file, transform='linear'):
    img = read_tif_file(tif_file)
    assert img is not None, 'tif file read error'
    assert hasattr(mod, transform)
    transform = getattr(mod, transform)
    img = transform(img)
    # convert to R G B I
    if img.ndim == 3:
        img = img[[2, 1, 0, 3]]
    cv2_save('/data/zfp/test.png', img)
    write_tif('/data/zfp/test.tif', img)
    plt.axis('off')
    if img.ndim == 3:
        img = img.transpose([1, 2, 0])
        plt.imshow(img[:, :, :3])
    else:
        plt.imshow(img, cmap='gray')
    plt.show()

def find_item_by_id(output_dir, id):
    files = json_load(os.path.join(output_dir, 'filelist.json'))
    for k, v in files.items():
        if v['id'] == id:
            print(k)
            print(v)


if __name__ == '__main__':
    pan_file = '/data/zfp/data/济南大学/高分图像/山东省1/山东省1/山东省1.tif'
    ms_file = '/data/zfp/data/济南大学/高分图像/多光谱/山东省1.tif'
    fusion_file = '/data/zfp/data/济南大学/融合图像/山东省1.tif'
    check_files = [pan_file, ms_file, fusion_file]
    check_files = [fusion_file]
    for tif_file in check_files:
        tif_file = tif_file.replace('山东省1', '山东省2')
        assert os.path.exists(tif_file) 
        gdal_show(tif_file, transform='linear')