#coding:utf-8
import os
import math
import numpy as np
import cv2
import argparse
from rknn.api import RKNN
from math import sqrt

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

np.set_printoptions(threshold=np.inf)


def priorBox():    
    num_priors = cfg.NUM_PRIORS
    image_size = cfg.INPUT_DIMS[1]
    feature_maps = cfg.FEATURE_MAPS
    min_sizes = cfg.MIN_SIZES
    max_sizes = cfg.MAX_SIZES
    steps = cfg.STEPS
    aspect_ratios = cfg.ASPECT_RATIOS

    mean = []

    for k in range(num_priors):
        f = feature_maps[k]
        for i in range(f):
            for j in range(f):
                f_k = image_size / steps[k]

                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = min_sizes[k] / image_size

                mean.append(cx)
                mean.append(cy)
                mean.append(s_k)
                mean.append(s_k)


                # aspect_ratio: 1
                s_k_prime = sqrt(s_k * (max_sizes[k] / image_size))

                mean.append(cx)
                mean.append(cy)
                mean.append(s_k_prime)
                mean.append(s_k_prime)

                # rest of aspect ratios

                # for ar in aspect_ratios[k]:
                for ll in range(len(aspect_ratios[k])):
                    ar = aspect_ratios[k][ll]
                    mean.append(cx)
                    mean.append(cy)
                    mean.append(s_k * sqrt(ar))
                    mean.append(s_k / sqrt(ar))

                    mean.append(cx)
                    mean.append(cy)
                    mean.append(s_k / sqrt(ar))
                    mean.append(s_k * sqrt(ar)) 
    
    #print(len(mean)/4.)
    return mean


def IntersectBBox(box1, box2):
    if box1[0]>box2[2] or box1[2]<box2[0] or box1[1]>box2[3] or box1[3]<box2[1]:
        return 0
    else:
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) 

        xx1 = max(box1[0], box2[0])
        yy1 = max(box1[1], box2[1])
        xx2 = min(box1[2], box2[2])
        yy2 = min(box1[3], box2[3])

        w = max(0, xx2-xx1)
        h = max(0, yy2-yy1)

        ovr = w*h / (area1 + area2 - w*h)
        return ovr


def ssd_post_process(conf_data, loc_data, image_draw):

    prior_data = priorBox()

    prior_bboxes = prior_data[:len(loc_data)]

    prior_num = int(len(loc_data) / 4) # 8732
    print('prior_num: ',prior_num)
    
    conf_data = conf_data.reshape(-1,cfg.NUM_CLS)
    print("conf_data shape: ",conf_data.shape)
   
    idx_class_conf = []
    bboxes = [] 
    
    # conf
    for prior_idx in range(0,prior_num):
        max_val = np.max(conf_data[prior_idx])
        max_idx = np.argmax(conf_data[prior_idx])
        if max_val > cfg.CONF_THRES and max_idx != 0:
            idx_class_conf.append([prior_idx, max_idx, max_val])
    
            
    # boxes
    for i in range(0,prior_num):
        bbox_center_x = 0.1 * loc_data[4*i+0] * prior_bboxes[4*i + 2] + prior_bboxes[4*i]
        bbox_center_y = 0.1 * loc_data[4*i+1] * prior_bboxes[4*i + 3] + prior_bboxes[4*i + 1]
        bbox_w        = math.exp(0.2 * loc_data[4*i+2]) * prior_bboxes[4*i + 2]
        bbox_h        = math.exp(0.2 * loc_data[4*i+3]) * prior_bboxes[4*i + 3]
        
        tmp = []
        tmp.append(max(min(bbox_center_x - bbox_w / 2., 1), 0))
        tmp.append(max(min(bbox_center_y - bbox_h / 2., 1), 0))
        tmp.append(max(min(bbox_center_x + bbox_w / 2., 1), 0))
        tmp.append(max(min(bbox_center_y + bbox_h / 2., 1), 0))
        bboxes.append(tmp)
    
    #print(len(idx_class_conf))
    
    #nms
    cur_class_num = 0
    idx_class_conf_ = []
    for i in range(0, len(idx_class_conf)): 
        keep = True
        k = 0
        while k < cur_class_num:
            if keep:
                ovr = IntersectBBox(bboxes[idx_class_conf[i][0]], bboxes[idx_class_conf_[k][0]])
                if idx_class_conf_[k][1]==idx_class_conf[i][1] and ovr > cfg.NMS_THRES:
                    if idx_class_conf_[k][2]<idx_class_conf[i][2]:
                        idx_class_conf_.pop(k)
                        idx_class_conf_.append(idx_class_conf[i])
                    keep = False
                    break
                k += 1
            else:
                break
        if keep:
            idx_class_conf_.append(idx_class_conf[i])
            cur_class_num += 1
                                                                  
    #print(idx_class_conf_)
       
    box_class_score = []     
   
    for i in range(0, len(idx_class_conf_)):
        bboxes[idx_class_conf_[i][0]].append(idx_class_conf_[i][1])
        bboxes[idx_class_conf_[i][0]].append(idx_class_conf_[i][2])
        box_class_score.append( bboxes[idx_class_conf_[i][0]])
        

    img = cv2.resize(image_draw, (cfg.INPUT_DIMS[2], cfg.INPUT_DIMS[1]))
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.load_default()
    
    for i in range(0, len(box_class_score)):
        x1 = int(box_class_score[i][0]*img.shape[1])
        y1 = int(box_class_score[i][1]*img.shape[0])
        x2 = int(box_class_score[i][2]*img.shape[1])
        y2 = int(box_class_score[i][3]*img.shape[0])
        color = (0, int(box_class_score[i][4]/20.0*255), 255)
        draw.line([(x1, y1), (x1, y2), (x2 , y2),
             (x2 , y1), (x1, y1)], width=2, fill=color)
        display_str = cfg.CLASSES[box_class_score[i][4]] +  ":" + str(box_class_score[i][5])
        display_str_height = np.ceil((1 + 2 * 0.05) * font.getsize(display_str)[1])+1
        
        if y1 > display_str_height:
            text_bottom = y1
        else:
            text_bottom = y1 + display_str_height
        
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(x1, text_bottom-text_height-2*margin), (x1+text_width, text_bottom)],fill=color) 
        draw.text((x1+margin, text_bottom-text_height-margin), display_str, fill='black', font=font)

    np.copyto(img, np.array(img_pil))
    cv2.imwrite("result.jpg", img)

def landmark_post_process(eyePts, mouthPts, img_ori):
    h,w = img_ori.shape[0:2]

    # eyePts = eyePts * 128
    eyePts = eyePts.reshape((11,2))
    eyePts[:,0] *= w
    eyePts[:,1] *= h
    for i in range(eyePts.shape[0]):
        cv2.circle(img_ori,(int(eyePts[i][0]),int(eyePts[i][1])),2,(255,0,0),-1)
        
    mouthPts = mouthPts * 128
    mouthPts = mouthPts.reshape((8,2))
    for i in range(mouthPts.shape[0]):
        cv2.circle(img_ori,(int(mouthPts[i][0]),int(mouthPts[i][1])),2,(255,0,0),-1)

    cv2.imwrite('result.jpg',img_ori)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('ssd', 'cnn'), help="mode:'cnn' or 'ssd'")
    parser.add_argument('-l','--landmark_demo', type=bool, default=False, help="run landmark demo and save pic")
    args = parser.parse_args()
    if args.mode == 'ssd':
        import config_ssd as cfg
    elif args.mode == 'cnn':
        import config_cnn as cfg
    else:
        print('invalid mode({})'.format(args.mode))

    # Create RKNN object
    rknn = RKNN()
    
    # pre-process config
    print('--> config model')
    rknn.config(mean_values=cfg.MEAN_VALUES, std_values=cfg.STD_VALUES, target_platform='RK3588')
    print('done')
  
    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_caffe(model=cfg.CAFFE_PROTO,
                          blobs=cfg.CAFFE_MODEL)
    if ret != 0:
        print('Load model failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=cfg.QUANT_IMGLIST)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(cfg.RKNN_OPTPATH)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    # TEST OPTS   
    # Set inputs
    img = cv2.imread(cfg.TEST_IMGPATH)
    img_draw = img.copy()
    img = cv2.resize(img, (cfg.INPUT_DIMS[2], cfg.INPUT_DIMS[1]))
    if cfg.INPUT_TYPE == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif cfg.INPUT_TYPE == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:,:,np.newaxis]

    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('rknn has {} output'.format(len(outputs)))
    print('done')

    # Accuracy analysis
    print('--> Accuracy analysis')
    ret = rknn.accuracy_analysis(inputs=[cfg.TEST_IMGPATH], output_dir='./snapshot')
    if ret != 0:
        print('Accuracy analysis failed!')
        exit(ret)
    print('done')

    if cfg.CHOOSE_MODE == 'SSD':
        print('--> Test RKNN SSD output')
        outputs[0] = outputs[0].reshape((-1, 1))
        outputs[1] = outputs[1].reshape((-1, 1))
        ssd_post_process(outputs[1], outputs[0],img_draw)
        print('done, output image is result.jpg')
    elif cfg.CHOOSE_MODE == 'CNN' and args.landmark_demo:
        print('--> Test landmark demo output')
        outputs[0] = outputs[0].reshape((-1, 1))
        outputs[1] = outputs[1].reshape((-1, 1))
        landmark_post_process(outputs[0], outputs[1], img_draw)
        print('done, output image is result.jpg')

    print('--> Log output')
    sim_dir = 'sim_opt'
    if not os.path.exists(sim_dir):
        os.system('mkdir -p ' + sim_dir)

    for idx,opt in enumerate(outputs):
        tmp = opt.flatten()
        np.save("{}/rknn_{}_output{}_shape_{}.npy".format(sim_dir,cfg.PRJ_NAME,idx,tmp.shape[0]),opt)
    print('done, we have save {} output as ndarray.'.format(len(outputs)))


    rknn.release()