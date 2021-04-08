import os
import glob
def remove_redundant_pth(input_dir):
    print(input_dir)
    pth_files = glob.glob(os.path.join(input_dir, 'epoch_*.pth'))
    files = []
    max_id = 0
    max_id_file = None
    for pth_file in pth_files:
        file_id = int(os.path.splitext(os.path.basename(pth_file))[0].split('_')[1])
        if file_id > max_id:
            if max_id_file is not None:
                files.append(max_id_file)
            max_id_file = pth_file
            max_id = file_id
        else:
            files.append(pth_file)
    for file in files:
        os.remove(file)

root = '/data/zfp/mmdetection/experiments/result'
dirs = os.listdir(root)
for pth_dir in dirs:
    remove_redundant_pth(os.path.join(root, pth_dir))