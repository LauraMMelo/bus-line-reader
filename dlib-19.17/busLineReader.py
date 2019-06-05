import cv2
import pytesseract as pyOcr
from PIL import Image
import numpy as np
from detector import segmentar_letreiro
import difflib


img = cv2.imread('./bus_examples/onibus-extra-fortaleza.jpg')


img = segmentar_letreiro(img)[0]
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = img[:,:,2]
cv2.imshow('ImagemP&B', img)

kernel = np.ones((5,5),np.uint8)

ret, threshImg = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)


threshImg = cv2.resize(threshImg, (750, 150))

threshImg = cv2.morphologyEx(threshImg, cv2.MORPH_CLOSE, kernel)

threshImg = cv2.bitwise_not(threshImg)


cv2.imshow('ImagemBinarizada', threshImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

binaryImg = threshImg
linha = pyOcr.image_to_string(binaryImg, lang='por')

print(linha)

list_of_lines = ['029 PARANGABA NAUTICO',
                 '028 ANTONIO BEZERRA PAPICU',
                 '401 MONTESE PARANBAGA',
                 '604 DIAS MACEDO CENTRO']

p = difflib.get_close_matches(linha.upper(), list_of_lines, n = 1, cutoff = 0.2)

print(p)