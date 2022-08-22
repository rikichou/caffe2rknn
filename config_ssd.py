# coding:utf-8
# COMMON SETS
PRJ_NAME = 'rm_ssd'
INPUT_DIMS = [3, 300, 300] #输入数据维度，CHW
INPUT_TYPE = 'BGR' #输入数据类型，['RGB', 'BGR', 'GRAY']
MEAN_VALUES = [104, 117, 123]#均值
STD_VALUES = [1, 1, 1] #标准差
CAFFE_MODEL = 'model_caffe/SSD_RFB_4class.caffemodel'#caffemodel文件
CAFFE_PROTO = 'model_caffe/SSD_RFB_4class.prototxt'#prototxt文件
QUANT_IMGLIST = 'ref_img_list.txt'#用于量化的txt索引
RKNN_OPTPATH = 'model_rknn/deploy_{}.rknn'.format(PRJ_NAME)#输出量化模型名称

# TEST SETS
TEST_IMGPATH = './test_img/road_300x300.jpg'
CHOOSE_MODE = 'SSD'#需要转换模型类型，['CNN', 'SSD']

## TEST FOR SSD
## SSD测试会将输出结果直接贴在原图上
CLASSES = ('__background__', 'neg', 'car', 'person')
NUM_CLS = len(CLASSES)
CONF_THRES = 0.5
NMS_THRES = 0.45
NUM_PRIORS = 6
VARIANCES = [0.1, 0.2]

FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
MIN_SIZES = [30, 60, 111, 162, 213, 264]
MAX_SIZES = [60, 111, 162, 213, 264, 315]
STEPS = [8, 16, 32, 64, 100, 300]
ASPECT_RATIOS = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
