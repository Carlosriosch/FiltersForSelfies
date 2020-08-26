import cv2 as cv
import numpy as np
import os
# capturar imagen desde la webcam
cap = cv.VideoCapture(0)
import tensorflow as tf

model= tf.keras.models.load_model('../FiltersForSelfies/intento2/proyecto/modelos/detector_barbijo_20epochs.h5')


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

    elif pred[0][2]>pred[0][1]:
        print("Pongase barbijo!!")

    else :
        print("No veo nada")
    
    
    
    cv.imshow('Ttulo de la ventana', img)
    k = cv.waitKey(30)
    if k == 27: # ESC (ASCII)
        break

        
cap.release()
cv.destroyAllWindows()