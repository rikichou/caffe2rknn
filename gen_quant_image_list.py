import os
import glob

IMAGE_DIR = r'/zhourui/workspace/pro/source/caffe2rknn/config/call/data/quant_images_112'
out_file_path = r'/zhourui/workspace/pro/source/caffe2rknn/config/call/data/quant_images_112/list.txt'

images_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))

with open(out_file_path,'w') as fp:
    for imgpath in images_list:
        fp.write(os.path.basename(imgpath)+'\n')