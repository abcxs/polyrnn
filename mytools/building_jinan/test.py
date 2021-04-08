import cv2
path = '/data/zfp/data/jinan_1/fusion/val/JPEGImages/36_0_1872.tif'
from utils import read_tif_file
dataset = read_tif_file(path).transpose((1, 2, 0))
H, W, C = dataset.shape
print(dataset.max())
print(H, W, C, dataset.dtype)
res = cv2.resize(dataset, (W * 2, H * 2))
print(res.shape, res.dtype)
print(dataset[0:2, 0:2, 0])
print(res[0:2, 0:2, 0])