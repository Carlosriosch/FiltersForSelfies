import cv2 as cv
import numpy as np
from add_mask1 import agregar_imagen
import imageio
class Filtro (object ): 
    def __init__(self, imagen): 
        self.imagen=imagen
    
    def agregador_imagen(self, img, x, y, posicionx, posiciony): 
        mascara_con_resize=cv.resize(self.imagen,(50,50))
        agregar_imagen(img, mascara_con_resize, int(posicionx), int(posiciony))
