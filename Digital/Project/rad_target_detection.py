import numpy as np
import matplotlib.pyplot as plt
import cv2


def find_contours(img):
    '''

    :param img:
    :return:
    '''
    thresh = img.copy()
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # filter out short ones
    contours = [cnt for cnt in contours if len(cnt) > 10]
    return contours


def get_threshold(img):
    blur = cv2.bilateralFilter(img, 5, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 2)
    return thresh


if __name__ == '__main__':
    pass
