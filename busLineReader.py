import cv2
import numpy as np
import pytesseract as pyOcr
from PIL import Image

img = Image.open('papicu.jpg').convert('RGB')
arrayImg = np.asarray(img).astype(np.uint8)
arrayImg[:, :, 0] = 0 # zerar o canal RED 
arrayImg[:, :, 2] = 0 # zerar o canal BLUE

imCv = cv2.cvtColor(arrayImg, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imCv, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

binaryImg = Image.fromarray(thresh)
linha = pyOcr.image_to_string(binaryImg, lang='por')

cv2.waitkey(0)

print(linha)

