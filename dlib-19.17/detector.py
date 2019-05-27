import os
import sys
import glob
import numpy as np
import dlib

import matplotlib.pyplot as plt

detector = dlib.simple_object_detector("detector.svm") #carrega o detector já treinado

def detectar_letreiro(img):
    """ Aplica do detector de objeto na imagem
    - img: Imagem com letreiros a serem detectados
    retorna: bounding-boxes do letreiros na imagem
    """
    dets = detector(img)
    return dets

def segmentar_letreiro(img):
    """ Segmenta o letreiro na imagem com a bounding-box
    - img: Imagem com letreiros a serem segmentados
    retorna: array de imagens [indice, altura, largura, canais]
    """
    dets = detectar_letreiro(img)
    letreiros = []
    for k, d in enumerate(dets):
        seg_img = img[d.top():d.bottom(), d.left():d.right()]
        letreiros.append(seg_img)
    
    return np.array(letreiros)

