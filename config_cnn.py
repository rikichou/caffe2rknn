# coding:utf-8
# COMMON SETS
PRJ_NAME = 'rm_smoke'
INPUT_DIMS = [1, 256, 256] #输入数据维度，CHW
INPUT_TYPE = 'GRAY' #输入数据类型，['RGB', 'BGR', 'GRAY']
MEAN_VALUES = [123.675]#均值
STD_VALUES = [58.395] #标准差
CAFFE_MODEL = 'config/smoke/model/20220628_256x256/deploy.caffemodel'#caffemodel文件
CAFFE_PROTO = 'config/smoke/model/20220628_256x256/deploy.prototxt'#prototxt文件
QUANT_IMGLIST = 'config/smoke/data/quant_images_256/image_list.txt'#用于量化的txt索引，注意txt内路径需写成相对路径，如本案例中wk0301/0.jpg
RKNN_OPTPATH = 'config/smoke/model/20220628_256x256/dms_smoke_3588.rknn'#输出量化模型名称

# TEST SETS
TEST_IMGPATH = 'config/smoke/data/test_img/00004.jpg'#指定测试图片
CHOOSE_MODE = 'CNN'#需要转换模型类型，['CNN', 'SSD']

