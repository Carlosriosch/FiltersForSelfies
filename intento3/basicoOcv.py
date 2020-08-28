import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from func_add_mask import agregar_imagen
from filtro import Filtro
import imageio


cap = cv.VideoCapture(0) 

model= tf.keras.models.load_model('../FiltersForSelfies/intento3/modelos/modelo_intento3_8epochs.h5')
detector_caras = cv.CascadeClassifier('../FiltersForSelfies/OpenCVFilter/haarcascade_frontalface_default.xml')



mascara_exito=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/exito.png')
mascara_precaucion=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/precaucion.png')
#mascara_=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/precaucion.png')

filtros=[Filtro(mascara_exito, -25, 30, 25), Filtro(mascara_precaucion, -25, 30, 25)]



while True:
    ret, img = cap.read() # leer la webcam
    img = cv.flip( img, 1 ) # flip horizontal para que sea un espejo
    
    img_gris=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    caras = detector_caras.detectMultiScale(img_gris, 1.3, 5)
    
    for i, (xc,yc,wc,hc) in enumerate(caras): 
        ##TODO arreglar las coordenadas de las máscaras
        
        cv.rectangle(img, (xc,yc), (xc+wc,yc+hc), (200,55,32))

        img_h, img_w = img.shape[:2]
        x = xc
        y = yc
        w = wc
        h = hc
    
    
        ejemplo = img[y:y+h, x:x+w] #recortar el rectangulo que nos interesa
        #cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255))
        


        cuad_chico = cv.resize(ejemplo, (256, 256))
                    # transformar el array a una dimensionalidad compatible con el modelo
        #cara_input = np.array([cara_chica.reshape(96, 96, 1)]) / 255

        cara_input =np.array([cuad_chico.reshape(256,256,3)])            # predecir
        
        pred=model.predict(cara_input)
        #print(pred)
        if pred[0]>0.3: #>pred[0][1]:
            print("Estás con barbijo!")
            filtros[0].agregar_a_imagen(img, 50, 50, 80, 80)
        else:     #if pred[0][1]>pred[0][0]:
            print("Pongase barbijo!!")
            filtros[1].agregar_a_imagen(img, 50, 50, 80, 80)
        #else :
            #print("No veo nada")
            #filtros[2].agregar_a_imagen(img, 50, 50, 80, 80)
        
        
    
    cv.imshow('Ttulo de la ventana', img)
    k = cv.waitKey(30)
    if k == 27: # ESC (ASCII)
        break

        
cap.release()
cv.destroyAllWindows()