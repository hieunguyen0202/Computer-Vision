import cv2
import numpy as np
from scipy.spatial.distance import cdist
import math


def find_features(img_org, img_contour):
    # find and contours
    contours, _ = cv2.findContours(img_contour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img_org, [approx], 0, (255, 0, 0), 1)

    # choose max contour is object
    compare = []
    for i in range(len(contours)):
        compare.append(len(contours[i]))
    max_ct = np.argmax(compare)
    cor = contours[max_ct]
    cor = cor[:, 0, :]

    # find vertical distance of object
    minmax = []
    for i in range(cor.shape[0]):
        minmax.append(cor[i][1])
    minmax = np.array(minmax)
    up_point = cor[minmax == min(minmax), :]
    down_point = cor[minmax == max(minmax), :]

    len_vertical = abs(up_point[0, 1]-down_point[0, 1])

    # find horizontal distance of object
    minmax = []
    for i in range(cor.shape[0]):
        minmax.append(cor[i][0])
    minmax = np.array(minmax)
    left_point = cor[minmax == min(minmax), :]
    right_point = cor[minmax == max(minmax), :]

    len_horizontal = abs(right_point[0, 0]-left_point[0, 0])

    # split object to 3
    cor1, cor2, cor3 = [], [], []
    # toi uu cac diem xet la diem center
    if len_vertical > len_horizontal:
        split_lines = len_vertical/3
        yline1 = up_point[0, 1] + split_lines
        yline2 = up_point[0, 1] + 2 * split_lines
        xline1 = left_point[0, 0]
        xline2 = right_point[0, 0]

        # p[1] is y coordinate
        for p in cor:
            if p[1] <= int(yline1):
                cor1.append(p)
            elif p[1] <= int(yline2):
                cor2.append(p)
            else:
                cor3.append(p)
    else:
        split_lines = len_horizontal / 3
        xline1 = left_point[0, 0] + split_lines
        xline2 = left_point[0, 0] + 2 * split_lines
        yline1 = up_point[0, 1]
        yline2 = down_point[0, 1]

        # p[0] is x coordinate
        for p in cor:
            if p[0] <= int(xline1):
                cor1.append(p)
            elif p[0] <= int(xline2):
                cor2.append(p)
            else:
                cor3.append(p)

    cv2.line(img_org, (int(xline1), int(yline1)), (int(xline2), int(yline1)), (0, 255, 0), thickness=1)
    cv2.line(img_org, (int(xline1), int(yline2)), (int(xline2), int(yline2)), (0, 255, 0), thickness=1)
    cv2.line(img_org, (int(xline1), int(yline1)), (int(xline1), int(yline2)), (0, 255, 0), thickness=1)
    cv2.line(img_org, (int(xline2), int(yline1)), (int(xline2), int(yline2)), (0, 255, 0), thickness=1)

    # contour of 3 parts of object
    cor1 = np.array(cor1)
    cor2 = np.array(cor2)
    cor3 = np.array(cor3)

    # find center
    arr = []
    center = []
    for i in range(int(yline1), int(yline2)):
        for j in range(int(xline1), int(xline2)):
            if img_contour[i, j] == 255:
                D = cdist(cor, np.array([[i, j]]))
                D = np.square(D)
                D = np.sum(D)
                arr.append(D)
                center.append([i, j])
    center = np.array(center)
    center = center[arr == min(arr), :]
    cv2.circle(img_org, (center[0][0], center[0][1]), radius=1, color=(0, 0, 255), thickness=3)

    # tim cac dinh cua 3 phan tam anh
    D = cdist(cor1, center)
    maxp = np.argmax(D)
    hi_point = cor1[maxp, :]
    x = hi_point[0]
    y = hi_point[1]

    D = cdist(cor2, center)
    minp = np.argmin(D)
    lo_point = cor2[minp, :]
    x2 = lo_point[0]
    y2 = lo_point[1]

    D = cdist(cor3, center)
    maxp = np.argmax(D)
    hi_point = cor3[maxp, :]
    x1 = hi_point[0]
    y1 = hi_point[1]

    d1 = math.sqrt((x1-x)**2+(y1-y)**2)
    d2 = math.sqrt((x2-x)**2+(y2-y)**2)
    d3 = math.sqrt((x2-x1)**2+(y2-y1)**2)

    cv2.line(img_org, (x, y), (x1, y1), (0, 0, 255), thickness=1)
    cv2.line(img_org, (x, y), (x2, y2), (0, 0, 255), thickness=1)
    cv2.line(img_org, (x2, y2), (x1, y1), (0, 0, 255), thickness=1)

    return img_org, center, d1, d2, d3
