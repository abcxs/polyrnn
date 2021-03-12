from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    params = ['/home/zhoufeipeng/code/vertex/configs/polygon/polyrnn_r50_fpn_1x_building.py',
              '/home/zhoufeipeng/code/vertex/work_dirs/polyrnn_r50_fpn_1x_building/latest.pth']
    args = parser.parse_args(params)
    img = '/home/zhoufeipeng/code/vertex/data/building/yunnan/val/JPEGImages/yunnan_48_0_0_4096.png'
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, img)
    # show the results
    # show_result_pyplot(model, img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
