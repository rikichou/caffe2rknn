# 使用说明
运行环境在192.168.140.202服务器的`rk:rk3588`docker容器中

## 转模型工具
模型转换采用脚本`run_genmodel.py`，该转换脚本需在docker容器内运行。请参照`config_cnn.py`与`config_ssd.py`进行相关参数配置。
- 转ssd模型
`
python test.py ssd
`
- 转cnn模型
`
python test.py cnn
`
- 转示例中的landmark模型并保存图像
`
python test.py cnn -l True
`

## 文件夹说明
- `model_caffe`：该文件夹用于存放caffemodel及prototxt文件，具体请在config中配置。
- `model_rknn`：该文件夹用于存放生成的rknn模型，具体请在config中配置。
- `quant_img`：该文件夹用于存放量化图片，具体请在config中配置。
- `sim_opt`：改文件夹用于存放转模型后pc模拟输出结果，具体请在config中配置。
- `snapshot`：浮点和量化后结果的对比。