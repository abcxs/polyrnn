# %%
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    config = '/home/zhoufeipeng/code/vertex/configs/polygon/polyrnn_r50_fpn_1x_building.py'
    checkpoint = '/home/zhoufeipeng/code/vertex/work_dirs/polyrnn_r50_fpn_1x_building/latest.pth'
    device = 'cuda:0'
    score_thr = 0.6
    img = '/home/zhoufeipeng/code/vertex/data/building/yunnan_small/val/JPEGImages/yunnan_small_32_0_512_4096.png'
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device=device)
    # test a single image
    result = inference_detector(model, img)
    # show the results
    show_result_pyplot(model, img, result, score_thr=score_thr)


if __name__ == '__main__':
    main()

# %%
