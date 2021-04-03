import numpy as np
import sys
import itertools
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm
from colorsys import hsv_to_rgb


def binarize_image(img, d=50, sig1=100, sig2=100, b_size=13, c=5):
    """

    :param img:
    :return:
    """
    blur = cv2.bilateralFilter(img, d, sig1, sig2)  # d>=5, sigma values (10 is small, bigger the more effect)
    # blur = cv.GaussianBlur(img,(5,5),0)

    # plt.imshow(blur)
    # plt.show()

    binary_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, b_size, c)
    
    return binary_img


def contour_image(binary, thresh=15):
    """

    :param binary:
    :param thresh:
    :return:
    """
    contours, hierarchy = cv2.findContours(binary,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # filter out short ones
    contours = [cnt for cnt in contours if len(cnt) > thresh]
    return contours


def find_ellipses(conts):
    """

    :param conts:
    :return:
    """
    ellipses = []
    hulls = []
    # for each contour fit an ellipse
    print('[INFO] fitting ellipses to contours')
    for i, cnt in enumerate(conts):
        # get convex hull of contour
        hull = cv2.convexHull(cnt, returnPoints=True)
        # defects = cv2.convexityDefects(cnt, hull)

        # hulls must be greater or equal to 5 to be able to fit an ellipse to it
        if len(hull) >= 5:
            ellipse = cv2.fitEllipse(np.array(hull))
            ellipses.append(ellipse)
            hulls.append(hull)

    print('[INFO] found {} valid ellipses'.format(len(ellipses)))
    return ellipses, hulls


def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=3, lineType=cv2.LINE_8, shift=0):
    """

    :param img:
    :param center:
    :param axes:
    :param angle:
    :param startAngle:
    :param endAngle:
    :param color:
    :param thickness:
    :param lineType:
    :param shift:
    :return:
    """
    center = (
        int(round(center[0])),
        int(round(center[1]))
    )
    axes = (
        int(round(axes[0]) / 2),
        int(round(axes[1]) / 2)
    )
    # cv.ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]])
    cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)


def find_rad_targets(ellipses, epsilon=5, lower_thresh=3.5, upper_thresh=7):
    print('[INFO] filtering the suspected targets from ellipses')
    rad_targets = []
    ells = []
    # organize ellipses into one array
    for ell in ellipses:
        center = ell[0]
        axes = ell[1]
        angle = ell[-1]
        center = (
            int(round(center[0])),
            int(round(center[1]))
        )
        axes = (
            int(round(axes[0])),
            int(round(axes[1]))
        )

        # we want ellipses with small eccentricity
        # if 0.5 < axes[0] / axes[1] < 2:
        ells.append([center[0], center[-1], axes[0], axes[-1], angle])

    ells = np.vstack(ells)

    # create an object to contain every possible pair of ellipses
    ell_pairs = itertools.combinations(ells, 2)

    for ells in tqdm(list(ell_pairs)):
        ell1 = ells[0]
        ell2 = ells[1]
        center1 = np.array([ell1[0], ell1[1]])
        center2 = np.array([ell2[0], ell2[1]])

        axes1 = np.array([ell1[2], ell1[3]])  # first is minor axis, second is major axis
        axes2 = np.array([ell2[2], ell2[3]])

        # check if in fact ellipses are concentric
        if np.linalg.norm(np.abs(center1 - center2)) < epsilon:
            # if they are concentric than check to see if correct ratio applies between the two
            if axes1[-1] < axes2[-1]:
                if lower_thresh < (axes2[-1] / axes1[-1]) < upper_thresh:
                    rad_targets.append(np.vstack((ell1, ell2)))
            else:
                if lower_thresh < (axes1[-1] / axes2[-1]) < upper_thresh:
                    rad_targets.append(np.vstack((ell2, ell1)))

    print('[INFO] found {} suspected as RAD targets'.format(len(rad_targets)))
    return rad_targets


def val_at_ellipse_coord(img, ellipse, n=200):
    """

    :param img:
    :param ellipse:
    :param n:
    :return:
    """
    # retrieve current ellipse parameters
    x0, y0 = ellipse[0], ellipse[1]
    minor_axis, major_axis = ellipse[2], ellipse[3]
    theta = np.deg2rad(ellipse[-1] + 90)
    # make a linear space for angles around the ellipse
    angles = np.linspace(0, 2 * np.pi, n)[:, None]
    # create points around perimeter of given ellipse
    ellipse_points = np.hstack((major_axis / 2 * np.cos(angles), minor_axis / 2 * np.sin(angles)))
    # define rotation matrix for points
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # extract rotated and translated points
    rot_points = np.dot(R, ellipse_points.T).T  # rotation
    rot_points[:, 0] += x0  # translation
    rot_points[:, 1] += y0

    # round points to integers for pixel indexing
    rot_points = np.around(rot_points, 0).astype(int)

    try:
        img_values = img[rot_points[:, 1], rot_points[:, 0]]  # xs - rows, ys - cols
    except IndexError:
        # for when the ellipse index is out of img bounds
        print('suspected target is out of image bounds and cannot be used')
        return rot_points, angles, None

    # normalize values to 0 or 1
    # max_val = img_values.max()
    # img_values[img_values <= 0.25 * max_val] = 0
    # img_values[img_values > 0.25 * max_val] = 1

    return rot_points, angles, img_values


def find_rad_encoding(img, rad_target):
    """

    :param img:
    :param rad_target:
    :return:
    """
    outer, inner = rad_target.copy(), rad_target.copy()
    # create outer and inner ellipse for target where outer is 85% and inner is 60% in axis sizes
    outer[2] *= 0.85
    outer[3] *= 0.85
    inner[2] *= 0.6
    inner[3] *= 0.6

    # find image values around perimeter of outer and inner ellipses
    pnts_outer, angles_outer, img_val_outer = val_at_ellipse_coord(img, outer, n=150)

    if img_val_outer is None:
        pnts_outer = -1
        pnts_inner = -1
        encoding = -1
        print('target is out of img bounds, encoding cannot be determined')
        return pnts_outer, pnts_inner, encoding

    pnts_inner, angles_inner, img_val_inner = val_at_ellipse_coord(img, inner, n=150)

    # for debug
    # plt.scatter(pnts_inner[:, 0], pnts_inner[:, 1])
    # plt.scatter(pnts_outer[:, 0], pnts_outer[:, 1])

    # get the angles where the image value along the ellipse is 0
    #
    # min_angle = np.min(angles_outer[img_val_outer == 0])  # * 180 / np.pi)
    # # find the index of the smallest angle where this is true
    # # start = np.where(angles_outer * 180 / np.pi == min_angle)[0][0]
    # start = np.where(angles_outer == min_angle)[0][0]

    # find the index where the outer ellipse value is first to be 0 (black)
    start = np.argmax(img_val_outer == 0)
    # now roll the array so that it start at that index
    img_val_inner = np.roll(img_val_inner, -start)
    # now split that array into 12 nearly equally sized pieces
    img_val_inner_split = np.array_split(img_val_inner, 12)
    # the median value should be either 255 or 0, calculate the encoding
    for i, segment in enumerate(img_val_inner_split):
        if np.median(segment) == 255:
            img_val_inner_split[i] = '1'
        else:
            img_val_inner_split[i] = '0'
    encoding = ''.join(img_val_inner_split)

    if encoding.startswith('0'):
        # return False
        pnts_outer = -1
        pnts_inner = -1
        encoding = -1
    if encoding != -1:
        print('[INFO] for target center ({}, {}) the coding is {}'.format(outer[0], outer[1], encoding))
    else:
        print('[INFO] for target center ({}, {}) - NOT A TARGET'.format(outer[0], outer[1]))
    return pnts_outer, pnts_inner, encoding


def draw_targets(img, targets):
    " draws ellips around the targets"
    if isinstance(targets, pd.DataFrame):
        for i, row in targets.iterrows():
            ell1 = targets.loc[i, 'target'][0]
            ell2 = targets.loc[i, 'target'][1]
            code = targets.loc[i, 'code']
            # convert color to rgb
            # color = tuple(round(i * 255) for i in hsv_to_rgb(targets.loc[i,'color'][0],targets.loc[i,'color'][1],targets.loc[i,'color'][2]))
            # draw_ellipse(img, (ell1[0], ell1[1]), (ell1[2], ell1[3]), ell1[-1], 0, 360, color)
            # draw_ellipse(img, (ell2[0], ell2[1]), (ell2[2], ell2[3]), ell2[-1], 0, 360, color)
            # for ell in targets:
            # ell1 = ell[0]
            # ell2 = ell[1]

            draw_ellipse(img, (ell1[0], ell1[1]), (ell1[2], ell1[3]), ell1[-1], 0, 360, (255, 0, 0))
            draw_ellipse(img, (ell2[0], ell2[1]), (ell2[2], ell2[3]), ell2[-1], 0, 360, (0, 0, 255))
            cv2.putText(img, code, (int(ell1[0]+2), int(ell1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        for tar in targets:
            ell1 = tar[0]
            ell2 = tar[1]
            draw_ellipse(img, (ell1[0], ell1[1]), (ell1[2], ell1[3]), ell1[-1], 0, 360, (255, 0, 0))
            draw_ellipse(img, (ell2[0], ell2[1]), (ell2[2], ell2[3]), ell2[-1], 0, 360, (0, 0, 255))


def get_cmap_string(palette, domain):
    domain_unique = np.unique(domain)
    hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
    mpl_cmap = plt.get_cmap(palette, lut=len(domain_unique))

    def cmap_out(X, **kwargs):
        return mpl_cmap(hash_table[X], **kwargs)

    return cmap_out


def targets_encoding(binary_img, targets):
    targets_df = pd.DataFrame(columns=['target', 'code', 'color'])
    targets_df['target'] = targets
    # find target encoding using the external ellipse in every ellipse-pair target
    ext_ellipses = np.vstack(targets)[1::2]  # here we take every 2nd ellipse in the pairs since it is the external
    codes = []
    # plt.imshow(binary_img, cmap='gray')
    for elli in ext_ellipses:
        pnts_outer, pnts_inner, code = find_rad_encoding(binary_img, elli)
        codes.append(code)

    # plt.show()
    
    targets_df['code'] = codes
    targets_df = targets_df[targets_df['code'] != -1]
    return targets_df


if __name__ == '__main__':
    
    image = cv2.imread(r'.\table_targets\20210325_121543.jpg')
    # image = cv2.imread(r'.\Shaul_Car2\20210313_144448.jpg')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # rgb image
    # plt.imshow(image)
    # plt.show()
    rgb_img = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # gray image
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    # firstly we will threshold the image to make it binary
    binary_img = binarize_image(gray, d=20, sig1=125, sig2=125, b_size=29, c=9)  # 100,100,17,5
    plt.imshow(binary_img, cmap='gray')
    plt.show()

    # getting contours of the binary image
    conts = contour_image(binary_img, thresh=7)
    c_img = rgb_img.copy()
    cv2.drawContours(c_img, conts, -1, (255, 0, 0), 1)
    plt.imshow(c_img)
    plt.show()

    # next step is to fit ellipses to the contours and filter out ones that cannot be candidates for target
    ellipses, hulls = find_ellipses(conts)

    # find concentric ellipses, check to see if ratio applies between each pair of concentric ellipses
    rad_targets = find_rad_targets(ellipses, lower_thresh=3.4, upper_thresh=7.6)

    # coding each target by it's shape and returning in data frame
    targets = targets_encoding(binarize_image(gray, d=20, sig1=100, sig2=100, b_size=51, c=10), rad_targets)  # 20,75,75,11,2

    # color map for the data
    # cmap = get_cmap_string(palette='hsv', domain=targets['code'])
    # targets['color'] = targets['code'].apply(cmap)

    # drawing found targets on img
    draw_targets(rgb_img, targets)

    cv2.imwrite("20210325_121543_coded.jpg", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    plt.imshow(rgb_img)
    plt.show()

    # for ell in ext_ellipses:
    #     points = val_at_ellipse_coord(binary, rad_targets[1][-1], n=100)
    # points, _, imvals = val_at_ellipse_coord(binary, ext_ellipses[1], n=100)

    # example usage
    # plt.figure()
    # x = np.linspace(0, np.pi * 2, 100)
    # for i_name, name in enumerate(targets['color']):
    #     plt.plot(x, np.sin(x) / i_name, c=name)
    # plt.show()
