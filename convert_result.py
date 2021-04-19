import argparse
from tqdm import tqdm
import pickle
import cv2
from approx_poly import approx
import numpy as np
import pycocotools.mask as mask_util

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def get_args():
    parser = argparse.ArgumentParser('convert result')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--is_approx', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    input_file = args.input_file
    output_file = args.output_file
    is_approx = args.is_approx
    
    with open(input_file, 'rb') as f:
        results = pickle.load(f)
        
    results_out = []
    for i in tqdm(range(len(results))):
        result = results[i]
        if len(result[0][0]) == 0:
            results_out.append((result[0], [[]]))
            continue
        imgs = mask_util.decode(result[1][0])
        h, w, c = imgs.shape
        img_segm = []
        for j in range(c):
            img = imgs[:, :, [j]]
            contours, _ = findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                img_segm.append(result[1][0][j])
                continue
            contours = np.concatenate(contours, axis=0)
            contour = cv2.convexHull(contours)

            if is_approx:
                try:
                    contour_ = approx(contour.copy()).reshape(-1, 2)
                except:
                    img_segm.append(result[1][0][j])
                    continue
            else:
                contour_ = cv2.minAreaRect(contour)
                contour_ = cv2.boxPoints(contour_).astype(np.int32)
            temp_img = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(temp_img, [contour_], 1)
            temp_img = np.array(temp_img[:, :, None], order='F', dtype='uint8')
            img_segm.append(mask_util.encode(temp_img)[0])
        results_out.append((result[0], [img_segm]))
    with open(output_file, 'wb') as f:
        pickle.dump(results_out, f)

if __name__ == '__main__':
    main()