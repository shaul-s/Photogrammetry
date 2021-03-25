import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2
import rad_target_detection as rtd

def get_targets(img, contour_thresh=5, epsilon=5, lower_thresh=3.5, upper_thresh=7):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rgb image
    plt.imshow(img)
    plt.show()
    rgb_img = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # gray image
    plt.imshow(gray, cmap='gray')
    plt.show()

    # firstly we will threshold the image to make it binary
    binary_img = rtd.binarize_image(gray)
    plt.imshow(binary_img, cmap='gray')
    plt.show()

    # getting contours of the binary image
    contours = rtd.contour_image(binary_img, contour_thresh)
    c_img = rgb_img.copy()
    cv2.drawContours(c_img, contours, -1, (255, 0, 0), 1)
    plt.imshow(c_img)
    plt.show()

    # next step is to fit ellipses to the contours and filter out ones that cannot be candidates for target
    ellipses, hulls = rtd.find_ellipses(contours)
    for i, ellip in enumerate(ellipses):
        rtd.draw_ellipse(c_img, ellip[0], ellip[1], ellip[2], 0, 360, (0, 0, 255))
    plt.imshow(c_img)
    plt.show()

    # find concentric ellipses, check to see if ratio applies between each pair of concentric ellipses
    rad_targets = rtd.find_rad_targets(ellipses, epsilon, lower_thresh, upper_thresh)

    # coding each target by it's shape
    targets_df = rtd.targets_encoding(binary_img, rad_targets)

    # color map for the data
    cmap = rtd.get_cmap_string(palette='viridis', domain=targets_df['code'])
    targets_df['color'] = targets_df['code'].apply(cmap)

    # drawing found targets on img
    rtd.draw_targets(rgb_img, targets_df)
    plt.imshow(rgb_img)
    plt.show()

    return targets_df

if __name__ == '__main__':
    image = cv2.imread(r'Shaul_Car\20210313_144612.jpg')
    targets = get_targets(image)


