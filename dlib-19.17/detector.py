import numpy as np
import dlib


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

#def ler_linha(img):
#    ''' Ler o letreiro em uma imagem
#    - img: Imagem do letreiro previamente segmentada
#    retorna: O texto do letreiro como uma string
#    '''
#    
#    img = img[:,:, 2]
#    kernel = np.ones((7,7),np.uint8)
#    
#    ret, threshImg = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
#    
#    threshImg = cv2.resize(threshImg, (750, 150))
#    threshImg = cv2.morphologyEx(threshImg, cv2.MORPH_CLOSE, kernel)
#    
#    threshImg = cv2.bitwise_not(threshImg)
#    
#    binaryImg = threshImg
#    linha = pyOcr.image_to_string(binaryImg, lang='por')
#    
#    return linha
#    
#    