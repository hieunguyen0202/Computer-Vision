import cv2
import numpy as np
import imutils


def maxFilter(n, image):
    size = (n, n)
    shape = cv2.MORPH_RECT
    mask = cv2.getStructuringElement(shape, size)
    imgResult = cv2.dilate(image, mask)
    return imgResult


# laplacian kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])


def preprocessing_img(img):
    img = imutils.resize(img, width=443, height=591)
    img = img[0:590, 0:440]
    img_org = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 17)

    image_sharp = cv2.filter2D(src=img, ddepth=0, kernel=kernel)
    s, thresh_sharp = cv2.threshold(image_sharp, 16, 255, cv2.THRESH_BINARY)
    o, thresh_object = cv2.threshold(img, 88, 255, cv2.THRESH_BINARY)

    img_contour = thresh_object + thresh_sharp

    # xoa dom den trong dep
    img_contour = maxFilter(3, img_contour)
    return img_org, img_contour
