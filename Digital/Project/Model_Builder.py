import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2
import rad_target_detection as rtd

def get_targets(img, binary_thresh=15, epsilon=5, lower_thresh=3.5, upper_thresh=7):


if __name__ == '__main__':
    image = cv2.imread('Car-with-coded-targets.jpg', 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # rgb image
    rgb_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # gray image

    # firstly we will threshold the image to make it binary
    binary = rtd.binarize_image(gray)
    plt.imshow(binary)
    plt.show()

