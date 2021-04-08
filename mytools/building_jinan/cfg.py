
seed = 1234
percent = 0.7
input_dir = '/usr/local/app/ssdceph/users/roczhou/detection/BaiduPCS-Go-v3.6.2-linux-386/jinan'
output_dir = '/usr/local/app/ssdceph/users/roczhou/detection/mmd/data/jinan'
min_size = 8
scale = 1
crop_size = 512
overlap_size = 64
test_overlap = False
thresh = 0.5
visual = True
transform = 'linear'
convert_uint8 = True
assert transform in ['scale', 'linear']
filelist = None
