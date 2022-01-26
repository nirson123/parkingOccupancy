import cv2
import numpy as np
import pandas as pd


class TmpModel:
    @staticmethod
    def predict(photo: pd.Series) -> int:
        return np.random.randint(2)


def draw_rectangles_with_labels(img_path: str, camera_number: int, model):
    CORD_SCALE = 1000 / 2592
    img = cv2.imread(img_path)
    positions = []
    with open(f'../unlabeled data set/CNR-EXT_FULL_IMAGE_1000x750/camera{camera_number}.csv', 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            X, Y, W, H = (int(int(x) * CORD_SCALE) for x in line.split(',')[1:])
            positions.append((X, Y, X+W, Y+H))
            crop = img[Y:Y+H, X:X+W]
            data_for_model = pd.Series(cv2.resize(crop, (70, 70)).flatten())
            predication = model.predict(data_for_model)
            color = (0, 255, 0) if predication == 1 else (0, 0, 255)
            cv2.rectangle(img, (X, Y), (X + W, Y + H), color, 1)

    cv2.imshow('photo', img)
    cv2.waitKey(0)


def main():
    path = '../unlabeled data set/CNR-EXT_FULL_IMAGE_1000x750/FULL_IMAGE_1000x750/SUNNY/2016-01-13/camera7/2016-01-13_0822.jpg'
    draw_rectangles_with_labels(path, 7, TmpModel)


if __name__ == '__main__':
    main()
