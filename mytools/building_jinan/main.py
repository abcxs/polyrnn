import cfg
from gen_filelist import gen_fielist
from preprocess import preprocess
from txt2coco import txt2coco
from utils import json_load

def t1():
    files = json_load(os.path.join(cfg.output_dir, 'filelist.json'))
    for k, v in files.items():
        print(k)
        print(v)
        break

if __name__ == '__main__':
    gen_fielist(cfg)
    preprocess(cfg)
    t1()
    txt2coco(cfg)


