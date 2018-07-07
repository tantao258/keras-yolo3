from yolo3.utils import mkdir_for_newProject, voc_xml_to_txt, get_classes
from yolo3.convert import Darknet_to_Keras
from yolo3.yolo import *
from voc_annotation import convert_annotation

import shutil

def main():

    config = {
        "method": 2,            #  1. quick_start  2. train_new_model
        "model_type": 1,        #  1. yolo         2. tiny-yolo
        "classes_path": "model_data/voc_classes.txt",
    }

    def quick_start():
        # 检验yolo3 weights文件是否存在
        if not os.path.exists(os.path.join("/model_data", "yolov3.weights")):
            print("Download YOLOv3 weights to dir: [/model_data/]")

        if config["model_type"] == 1:
            # 转换预先训练好的yolo3 weights
            if not os.path.exists(os.path.join("/model_data", "yolo.h5")):
                Darknet_to_Keras(Darknet_config_path="yolov3.cfg",
                                 Darknet_weights_path="model_data/yolov3.weights",
                                 output_path="model_data/yolo.h5")

        if config["model_type"] == 2:
            if not os.path.exists(os.path.join("/model_data", "tiny-yolo.h5")):
                Darknet_to_Keras(Darknet_config_path="yolov3-tiny.cfg",
                                 Darknet_weights_path="model_data/yolov3.weights",
                                 output_path="model_data/tiny-yolo.h5")

        # 物体检测
        detect_img(YOLO(model_type=config["model_type"]))

    def train_new_model(data_path):

        # 根据项目名称创建相关文件夹
        prefix = "projects"
        project_name = "new_project"
        if not os.path.exists(os.path.join(prefix, project_name)):
            mkdir_for_newProject(project_name=project_name, prefix=prefix)
        print("Directories of Project:{} have been created!".format(project_name))

        # 复制图片到目标文件夹
        if len(os.listdir(os.path.join(prefix, project_name, "VOC2007", "JPEGImages"))) == 0:
            """
            os.path.join(prefix, project_name, "VOC2007", "JPEGImages")为空，将训练数据复制到该文件夹
            """
            count = 1
            for file in os.listdir(data_path):
                if len(str(count)) < 5:
                    new_name = "0" * (5 - len(str(count))) + str(count) + ".jpg"
                    shutil.copy(os.path.join(data_path, file),
                                os.path.join(prefix, project_name, "VOC2007", "JPEGImages", new_name))
                count += 1
            print("Train_data have been copied from: {} to {} "
                  .format(data_path, os.path.join(prefix, project_name, "VOC2007", "JPEGImages")))

        # 标注数据
        if len(os.listdir(os.path.join(prefix, project_name, "VOC2007", "JPEGImages"))) != 0 and \
                len(os.listdir(os.path.join(prefix, project_name, "VOC2007", "Annotations"))) == 0:
            print("Please annotate images of directory: {}"
                  .format(os.path.join(prefix, project_name, "VOC2007", "JPEGImages")))
        if len(os.listdir(os.path.join(prefix, project_name, "VOC2007", "JPEGImages"))) != 0 and \
                len(os.listdir(os.path.join(prefix, project_name, "VOC2007", "JPEGImages"))) != \
                len(os.listdir(os.path.join(prefix, project_name, "VOC2007", "Annotations"))) :
            a = input("Please verify Annotations have complete,(Y/N)?")
            if a == "N" or a == "n":
                print("please complete Annotations first.")
                exit()
        else:
            print("Annotations have complete！")

        # 生成VOC2007格式数据
        if len(os.listdir(os.path.join(prefix, project_name, "VOC2007", "Annotations"))) != 0:
            voc_xml_to_txt(xmlfilepath=os.path.join(prefix, project_name, "VOC2007", "Annotations"),
                           txtsavepath=os.path.join(prefix, project_name, "VOC2007", "ImageSets"),
                           trainval_percent=0.8,
                           train_percent=0.8)

        # VOC2007格式数据生成yolo3能用的txt数据
        # 获取类别信息
        sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
        classes = get_classes(classes_path=config["classes_path"])

        for year, image_set in sets:
            image_ids = open(
                '{}/{}/VOC{}/ImageSets/{}.txt'.format(prefix, project_name, year, image_set)).read().strip().split()
            list_file = open('{}/{}/{}.txt'.format(prefix, project_name, image_set), 'w')
            for image_id in image_ids:
                list_file.write('{}/{}/VOC{}/JPEGImages/{}.jpg'
                                .format(prefix, project_name, year, image_id))
                convert_annotation(prefix, project_name, year, image_id, list_file, classes)
                list_file.write('\n')
            list_file.close()

        # 训练模型

    if config["method"] == 1:
        quick_start()


    if config["method"] == 2:
        train_new_model(data_path="../dataset/")





main()

