import cv2
import pytesseract as pyOcr
from PIL import Image

img = cv2.imread('papicu.jpg', 0)

cv2.imshow('ImagemP&B', img)

ret, threshImg = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

cv2.imshow('ImagemBinarizada', threshImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

binaryImg = Image.fromarray(threshImg)
linha = pyOcr.image_to_string(binaryImg, lang='por')

print(linha)

