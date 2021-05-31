import os
from flask import Flask, request, render_template
from numpy.lib.polynomial import poly
from werkzeug.utils import secure_filename
import base64
import cv2
import numpy as np
from model import Model
import glob
import argparse
import tifffile as tiff

app = Flask(__name__)
app.config['UPLOAD_IMAGES'] = 'upload_images'
app.config['INFERENCE_RESULT'] = 'inference_output'
os.makedirs(app.config['UPLOAD_IMAGES'], exist_ok=True)
os.makedirs(app.config['INFERENCE_RESULT'], exist_ok=True)

ALLOW_IMAGE_EXT = ['.png', '.jpg', '.jpeg', '.tif']
MODEL_MAPPING = dict(
    poly=('./checkpoints/yunnan', './checkpoints/mask'),
    tf_poly=('./checkpoints/jinan', './checkpoints/tf_mask'),
    det=('./checkpoints/tf_net', )
)

CUR_MODEL_NAME = None

def parse_args():
    parser = argparse.ArgumentParser(description='inference demo')
    parser.add_argument('--model_name', help='model name', default='poly')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.5,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=5,
        help='number of gpu to use')
    parser.add_argument(
        '--port',
        type=int,
        default=36644,
        help='mapping port')
    args = parser.parse_args()
    return args

def allow_file(file):
    return os.path.splitext(file)[1].lower() in ALLOW_IMAGE_EXT

def img2stream(img_path):
    img_stream = ''
    with open(img_path, mode='rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
        img_stream = img_stream.decode('utf-8')
    return img_stream

@app.route('/')
def index():
    return render_template('main.html')

def dispatch_files(files):
    pan_file, ms_file, fusion_file = None, None, None
    for f in files:
        if 'pan' in f:
            pan_file = f
        elif 'ms' in f:
            ms_file = f
        elif 'fusion' in f:
            fusion_file = f
        else:
            raise ValueError('don\'t support')
    assert pan_file is not None and ms_file is not None and fusion_file is not None
    print(pan_file, ms_file, fusion_file)
    return pan_file, ms_file, fusion_file

def read_tif(tif_file):
    img = tiff.imread(tif_file)
    img = img[:, :, :3][:, :, ::-1]
    img = img.copy()
    return img

@app.route('/predict_box_tf', methods=['GET', 'POST'])
def predict_box_tf():
    if request.method == 'POST':
        if CUR_MODEL_NAME != 'det':
            init_model('det')
        files = request.files.getlist('file')

        if len(files) <= 0:
            return render_template('tf_det_base.html')

        input_paths = []
        for file in files:
            # secure_filename好像不能读取中文名
            filename = secure_filename(file.filename)
            print(f'读取文件：{filename}')
            if not allow_file(filename):
                print('格式不支持')
                continue
            file_path = os.path.join(app.config['UPLOAD_IMAGES'], filename)
            file.save(file_path)
            input_paths.append(file_path)
        pan_file, ms_file, fusion_file = dispatch_files(input_paths)
        bn = os.path.basename(fusion_file)

        src_img = read_tif(fusion_file)
        out_src_path = os.path.join(app.config['INFERENCE_RESULT'], os.path.splitext(bn)[0] + '_src.png')
        cv2.imwrite(out_src_path, src_img)

        result = model(pan_file)
        img = model.show_result(fusion_file, result, ret_det=True)
        out_det_path = os.path.join(app.config['INFERENCE_RESULT'], os.path.splitext(bn)[0] + '.png')
        cv2.imwrite(out_det_path, img)

        det_img_stream = img2stream(out_det_path)
        src_img_stream = img2stream(out_src_path)
        ret = dict(
            src_img_stream=src_img_stream,
            det_img_stream=det_img_stream
        )

        return render_template('tf_det.html', **ret)

    return render_template('tf_det_base.html')

@app.route('/predict_polygon_tf', methods=['GET', 'POST'])
def predict_polygon_tf():
    if request.method == 'POST':
        if CUR_MODEL_NAME != 'tf_poly':
            init_model('tf_poly')
        files = request.files.getlist('file')

        if len(files) <= 0:
            return render_template('tf_poly_base.html')

        input_paths = []
        for file in files:
            # secure_filename好像不能读取中文名
            filename = secure_filename(file.filename)
            print(f'读取文件：{filename}')
            if not allow_file(filename):
                print('格式不支持')
                continue
            file_path = os.path.join(app.config['UPLOAD_IMAGES'], filename)
            file.save(file_path)
            input_paths.append(file_path)
        pan_file, ms_file, fusion_file = dispatch_files(input_paths)
        bn = os.path.basename(fusion_file)

        src_img = read_tif(fusion_file)
        out_src_path = os.path.join(app.config['INFERENCE_RESULT'], os.path.splitext(bn)[0] + '_src.png')
        cv2.imwrite(out_src_path, src_img)

        result = model(pan_file)
        img, poly_point_num = model.show_result(fusion_file, result)
        out_poly_path = os.path.join(app.config['INFERENCE_RESULT'], 'poly_' + os.path.splitext(bn)[0] + '.png')
        cv2.imwrite(out_poly_path, img)

        result = mask_model(pan_file)
        img, mask_point_num = mask_model.show_result(fusion_file, result)
        out_mask_path = os.path.join(app.config['INFERENCE_RESULT'], 'mask_' + os.path.splitext(bn)[0] + '.png')
        cv2.imwrite(out_mask_path, img) 

        src_img_stream = img2stream(out_src_path)
        poly_img_stream = img2stream(out_poly_path)
        mask_img_stream = img2stream(out_mask_path)
        mask_point_num = round(mask_point_num, 2)
        poly_point_num = round(poly_point_num, 2)

        ret = dict(
            src_img_stream=src_img_stream,
            mask_img_stream=mask_img_stream, 
            poly_img_stream=poly_img_stream, 
            mask_point_num=mask_point_num, 
            poly_point_num=poly_point_num
        )

        return render_template('tf_ret.html', **ret)
        # output_paths.append(out_img_path)
    return render_template('tf_poly_base.html')

@app.route('/predict_polygon_rgb', methods=['GET', 'POST'])
def predict_polygon_rgb():
    if request.method == 'POST':
        if CUR_MODEL_NAME != 'poly':
            init_model('poly')

        files = request.files.getlist('file')

        if len(files) <= 0:
            return render_template('rgb_base.html')

        file = files[0]
        # secure_filename好像不能读取中文名
        filename = secure_filename(file.filename)
        print(f'读取文件：{filename}')
        if not allow_file(filename):
            print('格式不支持')
            return render_template('rgb_base.html')

        file_path = os.path.join(app.config['UPLOAD_IMAGES'], filename)
        bn = os.path.basename(file_path)
        file.save(file_path)

        result = model(file_path)
        img, poly_point_num = model.show_result(file_path, result)
        out_poly_path = os.path.join(app.config['INFERENCE_RESULT'], f'poly_{bn}')
        cv2.imwrite(out_poly_path, img)

        result = mask_model(file_path)
        img, mask_point_num = mask_model.show_result(file_path, result)
        out_mask_path = os.path.join(app.config['INFERENCE_RESULT'], f'mask_{bn}')
        cv2.imwrite(out_mask_path, img)

        poly_img_stream = img2stream(out_poly_path)
        mask_img_stream = img2stream(out_mask_path)
        src_img_stream = img2stream(file_path)
        mask_point_num = round(mask_point_num, 2)
        poly_point_num = round(poly_point_num, 2)

        ret = dict(
            src_img_stream=src_img_stream, 
            mask_img_stream=mask_img_stream, 
            poly_img_stream=poly_img_stream, 
            mask_point_num=mask_point_num, 
            poly_point_num=poly_point_num
        )
        
        
        return render_template('rgb_ret.html', **ret)
    return render_template('rgb_base.html')


def get_model_info(model_dir):
    config_file = glob.glob(os.path.join(model_dir, '*.py'))[0]
    checkpoint_file = glob.glob(os.path.join(model_dir, '*.pth'))[0]
    return config_file, checkpoint_file

def init_model(model_name):
    global model, mask_model
    model, mask_model = None, None
    CUR_MODEL_NAME = model_name
    model_dirs = MODEL_MAPPING[model_name]
    assert len(model_dirs) <= 2
    config_file, checkpoint_file = get_model_info(model_dirs[0])
    print('模型初始化', config_file, checkpoint_file)
    model = Model(config_file, checkpoint_file, args.score_thr, args.gpu_id)
    if len(model_dirs) > 1:
        config_file, checkpoint_file = get_model_info(model_dirs[1])
        print('模型初始化', config_file, checkpoint_file)
        mask_model = Model(config_file, checkpoint_file, args.score_thr, args.gpu_id)

if __name__ == '__main__':
    global args
    args = parse_args()
    print(args)
    init_model(args.model_name)
    
    app.run(host='0.0.0.0', port=args.port , debug=True)