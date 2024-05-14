# -*- coding: utf-8 -*-
"""
Created on Sun April 16 23:00:55 2024

@author: Bhavya Singh
"""
import numpy as np
import pickle
import cv2
from collections import deque


letter_count = {0: 'CHECK', 1: 'ka', 2: 'kha', 3: 'ga', 4: 'gha', 5: 'kna', 6: 'cha',
                    7: 'chha', 8: 'ja', 9: 'jha', 10: 'yna',
                    11: 'taamatar',
                    12: 'thaa', 13: 'daa', 14: 'dhaa', 15: 'adna', 16: 'tabala', 17: 'tha',
                    18: 'da',
                    19: 'dha', 20: 'na', 21: 'pa', 22: 'pha',
                    23: 'ba',
                    24: 'bha', 25: 'ma', 26: 'yaw', 27: 'ra', 28: 'la', 29: 'waw', 30: 'motosaw',
                    31: 'petchiryakha',32: 'patalosaw', 33: 'ha',
                    34: 'chhya', 35: 'tra', 36: 'gya', 37: 'CHECK'}


def keras_predict(model, image):
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), letter_count[pred_class]


def keras_process_image(img):
    image_x = 32
    image_y = 32
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

pickle_out = open("dict.pickle","wb")
pickle.dump(letter_count, pickle_out)
pickle_out.close()