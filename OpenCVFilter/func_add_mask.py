import numpy as np
import cv2 as cv
def agregar_imagen(fondo, imagen, x, y): 
    alto=imagen.shape[0]
    ancho=imagen.shape[1]
    cantidad_de_canales= imagen.shape[-1]
# Vamos a ver si la imagen tiene información de opacidad
    if cantidad_de_canales == 4:
        
        opacidad=imagen[:,:,3]/255
        opacidad_stack=np.stack([opacidad, opacidad, opacidad], axis=-1)
        #Alpha Blending

        #Generamos una imagen vacia
        im3ch=np.zeros((imagen.shape[0], imagen.shape[1], 3)) #imagen 3 canales
        #multiplicamos la opacidad a cada canal
        im3ch[:,:,0]=imagen[:,:,0]*opacidad
        im3ch[:,:,1]=imagen[:,:,1]*opacidad
        im3ch[:,:,2]=imagen[:,:,2]*opacidad
        #multiplicamos la imagen de fondo por  (1-la opacidad) y le sumamos la imagen con la info de la opacidad
        fondo[y:y+alto, x:x+ancho, :]= (1-opacidad_stack)*fondo[y:y+alto, x:x+ancho, :]+im3ch
    elif cantidad_de_canales == 3:
        fondo[y:y+alto, x:x+ancho, :] = imagen # si no hay info de opacidad reemplazamos la imagen como esta
    else: #asumir que la cantidad de canales es uno
        im3ch = cv.cvtColor(imagen, cv.COLOR_GRAY2BGR)
        fondo[y:y+alto, x:x+ancho, :] = im3ch 

