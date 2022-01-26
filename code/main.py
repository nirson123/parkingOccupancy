import os
from PIL import Image
import numpy as np
import pandas as pd
import csv
import cv2
from typing import Tuple

labels_path = '../CNR-EXT-Patches-150x150/LABELS/all.txt'
data_path = '../data.csv'
photos_path = '../CNR-EXT-Patches-150x150/PATCHES'


def build_data_set(labels_path:str, photos_path, dst: str, img_size: tuple[int, int]) -> None:

    labels_dic = build_labels_dic(labels_path)

    f = open(dst, 'w', newline='')
    writer = csv.writer(f)

    j = 0
    # lines = []
    for d1 in os.listdir(photos_path):
        for d2 in os.listdir(os.path.join(photos_path,d1)):
            for d3 in os.listdir(os.path.join(photos_path,d1,d2)):
                for file_name in os.listdir(os.path.join(photos_path,d1,d2,d3)):
                    if j % 1000 == 0:
                        print(f'photo {j}')
                    j += 1
                    img = cv2.imread(os.path.join(photos_path,d1,d2,d3,file_name))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, img_size)
                    line = img.flatten()
                    line = np.append(line, [labels_dic[file_name]]).astype(np.uint8)
                    writer.writerow(line)
    f.close()


def build_labels_dic(labels_path: str) -> dict:

    labels = {}

    with open(labels_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            name, label = line.split(' ')
            file_name = name.split('/')[-1]
            labels[file_name] = int(label)

    return labels


def main():
    build_data_set(labels_path, photos_path, data_path)


if __name__ == '__main__':
    main()
