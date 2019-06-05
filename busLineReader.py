import cv2
import pytesseract as pyOcr
from PIL import Image
import numpy as np

img = cv2.imread('teste2.jpg')
img = img[:,:,2]

cv2.imshow('ImagemP&B', img)
kernel = np.ones((7,7),np.uint8)

ret, threshImg = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)


threshImg = cv2.resize(threshImg, (750, 150))

threshImg = cv2.morphologyEx(threshImg, cv2.MORPH_CLOSE, kernel)

threshImg = cv2.bitwise_not(threshImg)


cv2.imshow('ImagemBinarizada', threshImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

binaryImg = threshImg
linha = pyOcr.image_to_string(binaryImg, lang='por')

print(linha)

