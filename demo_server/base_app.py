import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import base64
import cv2
import numpy as np
from model import Model
import argparse

app = Flask(__name__)
app.config['UPLOAD_IMAGES'] = 'upload_images'
app.config['INFERENCE_RESULT'] = 'inference_output'
os.makedirs(app.config['UPLOAD_IMAGES'], exist_ok=True)
os.makedirs(app.config['INFERENCE_RESULT'], exist_ok=True)

ALLOW_IMAGE_EXT = ['.png', '.jpeg', '.JPEG', '.PNG', '.JPG', '.tif']

def parse_args():
    parser = argparse.ArgumentParser(description='inference demo')
    parser.add_argument('--config', help='test config file path', default='checkpoints/yunnan/polyrnn_r50_fpn_1x_building_edge.py')
    parser.add_argument('--checkpoint', help='checkpoint file', default='checkpoints/yunnan/checkpoint.pth')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.5,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=4,
        help='number of gpu to use')
    args = parser.parse_args()
    return args

def allow_file(file):
    return os.path.splitext(file)[1] in ALLOW_IMAGE_EXT

def img2stream(img_path):
    img_stream = ''
    with open(img_path, mode='rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
        img_stream = img_stream.decode('utf-8')
    return img_stream

@app.route('/')
def index():
    return '<h1>Hello World</h1>'

@app.route('/predict_polygon_rgb', methods=['GET', 'POST'])
def predict_polygon_rgb():
    if request.method == 'POST':
        files = request.files.getlist('file')
        input_paths = []
        output_paths = []

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

            img = model(file_path)

            bn = os.path.basename(file_path)
            out_img_path = os.path.join(app.config['INFERENCE_RESULT'], bn)
            cv2.imwrite(out_img_path, img)
            output_paths.append(out_img_path)

        poly_img_stream = img2stream(output_paths[0])
        
        
        return render_template('base.html', img_stream=poly_img_stream)
    return render_template('base.html')



if __name__ == '__main__':
    global model
    args = parse_args()
    model = Model(args.config, args.checkpoint, args.score_thr, args.gpu_id)
  
    app.run(host='0.0.0.0', port=36644, debug=True)