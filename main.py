import os
from yolo3.utils import mkdir_for_newProject
from convert import Darknet_to_Keras
from yolo import *

def main():

    method = "train_new_model"
    model_type = "yolo"

    def quick_start(model_type="yolo"):
        # 检验yolo3 weights文件是否存在
        if not os.path.exists(os.path.join("/model_data", "yolov3.weights")):
            print("Download YOLOv3 weights to dir: [/model_data/]")

        if model_type == "yolo":
            # 转换预先训练好的yolo3 weights
            if not os.path.exists(os.path.join("/model_data", "yolo.h5")):
                Darknet_to_Keras(Darknet_config_path="yolov3.cfg",
                                 Darknet_weights_path="model_data/yolov3.weights",
                                 output_path="model_data/yolo.h5")

        if model_type == "tiny-yolo":
            if not os.path.exists(os.path.join("/model_data", "tiny-yolo.h5")):
                Darknet_to_Keras(Darknet_config_path="yolov3-tiny.cfg",
                                 Darknet_weights_path="model_data/yolov3.weights",
                                 output_path="model_data/tiny-yolo.h5")

        # 物体检测
        detect_img(YOLO(type=model_type))

    def train_new_model():
        prefix = "projects"
        project_name = "2"
        if not os.path.exists(os.path.join(prefix, project_name)):
            mkdir_for_newProject(project_name=project_name, prefix=prefix)

    if method == "quick_start":
        quick_start(model_type=model_type)
    if method == "train_new_model":
        train_new_model()





main()

