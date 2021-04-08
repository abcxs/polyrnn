'''
@Author: zfp
@Date: 2019-11-15 19:30:02
@LastEditTime : 2019-12-24 21:10:10
@FilePath: /mmdetection/mytools/tests/test.py
'''

# %%
from mytools.building_jinan.utils import read_pan_file
import numpy as np
import matplotlib.pyplot as plt
import cv2
pan_file = '/data/zfp/data/济南大学/高分图像/山东省1/山东省1/山东省1.tif'
pan_file = '/data/zfp/data/济南大学/高分图像/多光谱/山东省1.tif'
srcArray = read_pan_file(pan_file)

# %%
print(srcArray.shape)
print(srcArray.dtype)
print(srcArray.max(), srcArray.min())
a = srcArray[:3].transpose([1, 2, 0])
a = (a - a.min()) / (a.max() - a.min()) * 255
a = a.astype(np.uint8)
h, w, c = a.shape
print(h, w, c)
a = a.resize(a, (int(w * 2), int(h * 2)))
plt.imshow(a)
plt.show()
# %%
b = (srcArray - srcArray.min()) / (srcArray.max() - srcArray.min()) * 255
b = b.astype(np.uint8)

plt.imshow(b, cmap='gray')
plt.show()
# %%
import cv2
import numpy as np
a = np.random.rand(100, 100, 4).astype(np.int16)
b = cv2.resize(a, (200, 200))
print(b.shape, b.dtype)

# %%
pan_file = '/data/zfp/data/济南大学/高分图像/山东省1/山东省1/山东省1.tif'
ms_file = '/data/zfp/data/济南大学/高分图像/多光谱/山东省1.tif'
fusion_file = '/data/zfp/data/济南大学/融合图像/山东省1.tif'

import gdal
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from mytools.building_jinan.utils import gdal_show
gdal.SetConfigOption('SHAPE_ENCODING', "UTF8")
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
check_files = [pan_file, ms_file, fusion_file]
check_files = [fusion_file]
for tif_file in check_files:
    tif_file = tif_file.replace('山东省1', '山东省2')
    assert os.path.exists(tif_file) 
    dataset = gdal.Open(tif_file)
    srcArray = dataset.ReadAsArray()

    gdal_show(srcArray, transform='linear')
# %%
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
img = Image.open('/data/zfp/1.tif')
img = np.array(img)
print(img.shape, img.dtype, img.max(), img.min())
plt.imshow(img / 1024, cmap='gray')
plt.show()
# %%

# %%
import numpy as np
a = np.random.rand(2, 3)[...,None]
print(a)
b = np.tile(a, (1, 1, 3))
print(b.shape)
print(b)
# %%
import cv2
import matplotlib.pyplot as plt
import gdal
path = '/data/zfp/data/济南大学/融合图像/山东省1.tif'

im = gdal.Open(path).ReadAsArray()[:3]
im1 = im.reshape(3, -1)
min_v = im1.min(axis=-1)
max_v = im1.max(axis=1)
im = im.transpose((1, 2, 0))
img = (im - min_v) / (max_v - min_v) * 255
img = img.astype(np.uint8)
percent = 0.05
val = 256 * percent
small_id = img < val
large_id = img > (255 - val)
medium_id = ~(small_id | large_id)
img[small_id] = 0
img[large_id] = 255

img[medium_id] = (img[medium_id] - val) /(255-val -val) * 255
img = img.astype(np.uint8)
cv2.imwrite('/data/zfp/1.png', img[:, :, ::-1])
# plt.imshow(img)
# plt.show()
# %%
import gdal
import matplotlib.pyplot as plt
gdal.SetConfigOption('SHAPE_ENCODING', "UTF8")
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
dataset = gdal.Open('/data/zfp/1.tif')
srcArray = dataset.ReadAsArray()
srcArray = srcArray.transpose([1, 2, 0])
plt.imshow(srcArray[:, :, :3])
plt.show()

# %%
import matplotlib.pyplot as plt
from mytools.building_jinan.utils import read_tif_file
tif_file = '/data/zfp/data/jinan/pan/train/JPEGImages/0_0_0.tif'
srcArray = read_tif_file(tif_file)
plt.imshow(srcArray, cmap='gray')
plt.show()

# %%
import numpy as np
import cv2
img = np.random.rand(10, 10, 1)
img = cv2.resize(img, (20, 20))
print(img.shape)

# %%
from pycocotools.coco import COCO
path = '/data/zfp/data/jinan_2/pan/val/val.json'
coco = COCO(path)


# %%
imgIds = coco.getImgIds()
print(len(imgIds))
annsId = coco.getAnnIds()
print(len(annsId))
# %%
