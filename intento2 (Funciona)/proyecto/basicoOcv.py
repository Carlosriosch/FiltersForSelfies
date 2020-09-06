import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from func_add_mask import agregar_imagen
from filtro import Filtro
import imageio


cap = cv.VideoCapture(0)

model= tf.keras.models.load_model('../FiltersForSelfies/intento2/proyecto/modelos/detector_barbijo_20epochs.h5')



mascara_exito=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/exito.png')
mascara_precaucion=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/precaucion.png')
#mascara_=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/precaucion.png')

filtros=[Filtro(mascara_exito, -25, 30, 25), Filtro(mascara_precaucion, -25, 30, 25)]



while True:
    ret, img = cap.read() # leer la webcam
    img = cv.flip( img, 1 ) # flip horizontal para que sea un espejo
    
    
    img_h, img_w = img.shape[:2]
    x = img_w - 256
    y = 0
    w = 256
    h = 256
    
    
    ejemplo = img[y:y+h, x:x+w] #recortar el rectangulo que nos interesa
    cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255))
    


    cuad_chico = cv.resize(ejemplo, (256, 256))
                # transformar el array a una dimensionalidad compatible con el modelo
    #cara_input = np.array([cara_chica.reshape(96, 96, 1)]) / 255

    cara_input =np.array([cuad_chico.reshape(256,256,3)])            # predecir
    
    pred=model.predict(cara_input)
    #print(pred)
    if pred[0][0]>pred[0][2]:
        print("Estás con barbijo!")
        filtros[0].agregar_a_imagen(img, 50, 50, 80, 80)
    elif pred[0][2]>pred[0][1]:
        print("Pongase barbijo!!")
        filtros[1].agregar_a_imagen(img, 50, 50, 80, 80)
    else :
        print("No veo nada")
        #filtros[2].agregar_a_imagen(img, 50, 50, 80, 80)
    
    
    
    cv.imshow('Ttulo de la ventana', img)
    k = cv.waitKey(30)
    if k == 27: # ESC (ASCII)
        break

        
cap.release()
cv.destroyAllWindows()