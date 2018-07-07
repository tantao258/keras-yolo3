"""
在数据标注完成后，制作成VOC2007格式数据，
并转化为yolo3，能使用的数据，并保存在 "/create_train_data"文件夹下

"""
import xml.etree.ElementTree as ET


def convert_annotation(prefix, project_name, year, image_id, list_file, classes):
    in_file = open('{}/{}/VOC{}/Annotations/{}.xml'.format(prefix, project_name, year, image_id))
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