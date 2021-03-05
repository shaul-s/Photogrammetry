import cv2
import numpy as np
from matplotlib import pyplot as plt


def generalized_hough(img, template):
    """

    :param img: reference image
    :param template: target to detect
    :return: center point of detected object
    """
    ght = cv2.createGeneralizedHoughBallard()
    ght.setTemplate(template)
    [position, votes] = ght.detect(img)
    position, votes = position.squeeze(), votes.squeeze()

    # max_idx = np.argmax(votes, axis=0)
    # x, y = position[max_idx[0], :2]
    x, y = position[:2]

    return [x, y]


def hough_circles(img, vote_threshold=200, inner_threshold=55, minRadius=11, maxRadius=None):
    """
    hough circle detection implementation by shaul shmouelly & oren lauterman
    :param img: grayscale image 2d array
    :param vote_threshold: threshold for min votes to become a suspect of being center of circle
    :param inner_threshold: iner threshold for further filtration of pixel that are suspicious of being center of circles
    :param minRadius: the minimum radius value in pixel for which to start checking from
    :return: list of circle (x0, y0, radius) in image pixels
    """

    shape = img.shape
    # resizing for faster performance
    # new_shape = tuple([int(x / 5) for x in list(shape)])
    # new_shape = (510, 660)
    # img = cv2.resize(img, new_shape)

    gauss = cv2.GaussianBlur(img, (3, 3), 0)  # applying noise reduction
    canny = cv2.Canny(gauss, 75, 150)  # detecting circle edges
    # radius = int(max(shape[0] / 8, shape[1]) / 8)  # maxRadius value
    radius = maxRadius

    rows = img.shape[0]
    cols = img.shape[1]

    accumulator = np.zeros((radius, rows, cols))  # initializing accumulator array

    circle_indices = (canny == 255).nonzero()  # get indices of edges (in this case circles)

    x, y = circle_indices

    for r in range(minRadius, radius):  # start @ minRadius up to maxRadius (eighth of image size)
        for theta in range(0, 360):  # theta range
            a = (x - r * np.cos(theta * np.pi / 180)).astype(int)  # compute circle gradient
            b = (y - r * np.sin(theta * np.pi / 180)).astype(int)

            a_tmp = a[np.where((a > 0) & (a < rows) & (b > 0) & (b < cols))]  # filter irrelevant gradients
            b_tmp = b[np.where((b > 0) & (b < cols) & (a > 0) & (a < rows))]

            accumulator[r, a_tmp, b_tmp] += 1  # VOTE

        print('checked radius=' + str(r) + '[pixels]')

    # the inner threshold for filtering out results

    circles = []

    # now for every radius we will check if  votes are above the vote threshold
    # if it is we will further filter with the inner threshold and add to circle list
    for r, im in enumerate(accumulator):
        max_idx = np.where(im >= vote_threshold)
        if max_idx[0].size > 0:
            print('there are circles in radius={}'.format(r))
            im_tmp = im[max_idx]
            im_tmp = im_tmp[im_tmp > max(im_tmp) - inner_threshold]
            for center_value in np.unique(np.sort(im_tmp)):
                center = np.where(im == center_value)
                if center[0].size > 1:
                    for i, j in zip(center[0], center[1]):
                        circles.append(np.array([int(j), int(i), r]))
                else:
                    circles.append(np.array([int(center[1]), int(center[0]), r]))

    return circles


def plot_circles(ref_image, circles):
    for c in circles:
        # draw the outer circle
        cv2.circle(ref_image, (c[0], c[1]), c[2], (255, 0, 0), 2)
        # draw the center of the circle
        cv2.circle(ref_image, (c[0], c[1]), 2, (0, 255, 0), 2)

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(resized)
    plt.subplot(1, 2, 2)
    plt.title('Circles Detected')
    plt.axis('off')
    plt.imshow(circle_img)
    plt.show()


if __name__ == '__main__':
    # path = r'images\0001.jpg'
    # image = cv2.imread(path, 0)
    # # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(image, (510, 660))

    # IF NOT ALL WANTED CIRCLED DETECTED - NEED TO CHANGE INNER THRESHOLD
    # THIS PART FOR SELF IMPLEMENTED CIRCLE DETECTION
    # circles = hough_circles(resized, vote_threshold=170, inner_threshold=100, minRadius=11, maxRadius=80)
    # circles = np.vstack(circles)
    #
    # resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    # circle_img = resized.copy()
    #
    # plot_circles(circle_img, circles)

    # GENERAL HOUGH TRANSFORM - OPENCV
    template = cv2.imread(r'car_template.jpg', 0)

    ref_image = cv2.imread(r'Car-with-coded-targets.jpg')

    detected_target = generalized_hough(cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY), template)

    copy = ref_image.copy()
    cv2.circle(copy, (int(detected_target[0]), int(detected_target[1])), 50, (0, 0, 255), 100)

    copy = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(ref_image)

    plt.subplot(1, 3, 2)
    plt.title('Template')
    plt.axis('off')
    plt.imshow(template, cmap='gray')

    plt.subplot(1,3,3)
    plt.title('Target Detected')
    plt.axis('off')
    plt.imshow(copy)

    plt.show()


