# coding:utf-8
# COMMON SETS
PRJ_NAME = 'rm_call'
INPUT_DIMS = [1, 112, 112] #输入数据维度，CHW
INPUT_TYPE = 'GRAY' #输入数据类型，['RGB', 'BGR', 'GRAY']
MEAN_VALUES = [127]#均值
STD_VALUES = [1] #标准差
CAFFE_MODEL = 'config/call/model/20220711_dmsonly/deploy.caffemodel'#caffemodel文件
CAFFE_PROTO = 'config/call/model/20220711_dmsonly/deploy.prototxt'#prototxt文件
QUANT_IMGLIST = 'config/call/data/quant_images_112/list.txt'#用于量化的txt索引，注意txt内路径需写成相对路径，如本案例中wk0301/0.jpg
RKNN_OPTPATH = 'config/call/model/20220711_dmsonly/dms_call5Classify_3588.rknn'#输出量化模型名称

# TEST SETS
TEST_IMGPATH = 'config/call/data/test_images/113.jpg'#指定测试图片
CHOOSE_MODE = 'CNN'#需要转换模型类型，['CNN', 'SSD']

