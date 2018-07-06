import os
import sys
import subprocess
import shutil


if __name__ == '__main__':

    # 清空制作数据集相关文件夹
    top = "./train_data/voc2007"
    for root, dirs, files in os.walk(top=top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))

    # 复制图片到相关文件夹
    data_path = sys.argv[1]
    count = 1
    for file in os.listdir(data_path):
        if len(str(count)) < 5:
            new_name = "0"*(5-len(str(count))) + str(count) + ".jpg"
            shutil.copy(os.path.join(data_path, file), os.path.join(top, "JPEGImages", new_name))
        count += 1
    print("complete data copy!")
