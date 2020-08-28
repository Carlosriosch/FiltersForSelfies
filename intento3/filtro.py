import cv2 as cv
import numpy as np
import os
from func_add_mask import agregar_imagen
import imageio
class Filtro(object):
    def __init__(self, 
                imagen,
                offset_ancho,
                offset_x,
                offset_y):
        self.imagen = imagen
        self.offset_ancho = offset_ancho
        self.offset_x = offset_x
        self.offset_y = offset_y
    def agregar_a_imagen(self, img, x, y ,w, h): 
        mascara_chica = cv.resize(self.imagen, (w+ self.offset_ancho, w+ self.offset_ancho))
        agregar_imagen(img, mascara_chica, x, y)
         #con esto imprimimos cada captura de open cv en el while
    #open cv también nos permite hacer llamadas a eventos 