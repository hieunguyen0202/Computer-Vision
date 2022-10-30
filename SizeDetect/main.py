from find_feature import find_features
from preprocessing import preprocessing_img
import cv2

img = cv2.imread('train image/size16/dep16 (8).jpg')
img_org, img_contour = preprocessing_img(img)
img, center, d1, d2, d3 = find_features(img_org, img_contour)
print(center)
print(d1, d2, d3)

cv2.imshow('dep', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
