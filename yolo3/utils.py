"""Miscellaneous utility functions."""

import os
import random
from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


# 创建初始项目相关文件夹
def mkdir_for_newProject(prefix, project_name):
    os.mkdir(os.path.join(prefix, project_name))
    os.mkdir(os.path.join(prefix, project_name, "VOC2007"))
    if os.path.exists(os.path.join(prefix, project_name, "VOC2007")):
        os.mkdir(os.path.join(prefix, project_name, "VOC2007", "Annotations"))
        os.mkdir(os.path.join(prefix, project_name, "VOC2007", "ImageSets"))
    os.mkdir(os.path.join(prefix, project_name, "VOC2007", "JPEGImages"))
    os.mkdir(os.path.join(prefix, project_name, "logs"))
    os.mkdir(os.path.join(prefix, project_name, "checkpoint"))
    return None


# 将voc2007标注的xml数据转化为txt数据
def voc_xml_to_txt(xmlfilepath="create_train_data/VOC2007/Annotations",
                   txtsavepath ="create_train_data/VOC2007/ImageSets",
                   trainval_percent=0.8,
                   train_percent=0.8):

    """
    在标注完成之后的数据[*.xml]，生成ImageSets/Main中，test.txt/train.txt/trainval.txt/val.txt
    """

    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)

    # 训练集+验证集数量占总样本 trainval_percent
    tv = int(num * trainval_percent)
    # 训练集数量占 训练集+验证集数量 train_percent
    tr = int(tv * train_percent)

    list = range(num)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open(os.path.join(txtsavepath, "trainval.txt"), 'w')
    ftest = open(os.path.join(txtsavepath, "test.txt"), 'w')
    ftrain = open(os.path.join(txtsavepath, "train.txt"), 'w')
    fval = open(os.path.join(txtsavepath, "val.txt"), 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


# 根据txt文件获取物体检测类别
def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    """
    ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    """
    return class_names


# 根据txt文件获取 anchors
def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors_reshape = np.array(anchors).reshape(-1, 2)
    """
    [[ 10.  13.]
     [ 16.  30.]
     [ 33.  23.]
     [ 30.  61.]
     [ 62.  45.]
     [ 59. 119.]
     [116.  90.]
     [156. 198.]
     [373. 326.]]
    """
    return anchors_reshape


def compose(*funcs):
    """
    Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """

    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def letterbox_image(image, size):
    """
    resize image with unchanged aspect ratio using padding
    """

    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.

        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
