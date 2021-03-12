from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
config_file = '/home/zhoufeipeng/code/vertex/configs/polygon/mask_rcnn_r50_fpn_1x_building.py'
cfg = Config.fromfile(config_file)
dataset = build_dataset(cfg.data.train)
print(dataset.pipeline)
dataloader = build_dataloader(dataset, 1, 0, dist=False, seed=1234)
for data in dataloader:
    break
print(data.keys())
polygons = data['gt_polygons'].data
print('gpu数量：', len(polygons))
print('图片数量：', len(polygons[0]))
print('多边形数量：', len(polygons[0][0]))
for sample in polygons[0][0]:
    print(sample.shape)