# coding:utf-8
# COMMON SETS
PRJ_NAME = 'rm_landmark'
INPUT_DIMS = [1, 128, 128] #输入数据维度，CHW
INPUT_TYPE = 'GRAY' #输入数据类型，['RGB', 'BGR', 'GRAY']
MEAN_VALUES = [0]#均值
STD_VALUES = [255] #标准差
CAFFE_MODEL = 'model_caffe/multiout_2.caffemodel'#caffemodel文件
CAFFE_PROTO = 'model_caffe/multiout_2.prototxt'#prototxt文件
QUANT_IMGLIST = 'ref_img_list_landmark.txt'#用于量化的txt索引，注意txt内路径需写成相对路径，如本案例中wk0301/0.jpg
RKNN_OPTPATH = 'model_rknn/deploy_{}.rknn'.format(PRJ_NAME)#输出量化模型名称

# TEST SETS
TEST_IMGPATH = './test_img/11111.jpg'#指定测试图片
CHOOSE_MODE = 'CNN'#需要转换模型类型，['CNN', 'SSD']

