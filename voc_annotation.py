"""
在数据标注完成后，制作成VOC2007格式数据，
并转化为yolo3，能使用的数据，并保存在 "/create_train_data"文件夹下


"""

from os import getcwd
from yolo3.utils import voc_xml_to_txt, get_classes
import xml.etree.ElementTree as ET


def convert_annotation(year, image_id, list_file):
    in_file = open('create_train_data/VOC{}/Annotations/{}.xml'.format(year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == '__main__':
    # 获取类别信息
    sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
    classes = get_classes(classes_path="model_data/voc_classes.txt")

    # 将标注的xml数据转化为txt数据，构建voc2007数据集
    voc_xml_to_txt(xmlfilepath="create_train_data/VOC2007/Annotations",
                   txtsavepath="create_train_data/VOC2007/ImageSets/Main",
                   train_percent=0.9,
                   trainval_percent=0.1)

    # 将构建的voc2007数据集，转化为yolo3能用的txt文件
    for year, image_set in sets:
        image_ids = open('create_train_data/VOC{}/ImageSets/Main/{}.txt'.format(year, image_set)).read().strip().split()
        list_file = open('create_train_data/{}.txt'.format(image_set), 'w')
        for image_id in image_ids:
            list_file.write('{}/creat_train_data/VOC{}/JPEGImages/{}.jpg'.format(getcwd(), year, image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()

