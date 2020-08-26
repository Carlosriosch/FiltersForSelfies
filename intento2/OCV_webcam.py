import cv2 as cv 
import imageio
import os #util para manipular el sistema operativo 
from func_add_mask import agregar_imagen
from filtro import Filtro
import tensorflow as tf
mascara=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/carnaval.png')
mascara2=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/exito.png')
mascara2=cv.resize(mascara2,(25,25))
from tensorflow import keras
#mascara2=cv.flip(mascara, 0)
filtros=[Filtro(mascara, -25, 30, 25), Filtro(mascara2, -25, 30, 25)]
offset_ancho = -25
##Obs podriamos usar YOLO pero es una herramienta poderosa de más. Mejor vamos a usar una herramienta de OpenCv llamada
#HAAR Cascade Classifier que sirve solo para detectar rostros

detector_caras = cv.CascadeClassifier('../FiltersForSelfies/OpenCVFilter/haarcascade_frontalface_default.xml')
#el problema es que este detector funciona solo para un canal, entonces tenemos que unir los canales (preprocesarlos)
detector_de_barbijo = tf.keras.models.load_model('../FiltersForSelfies/intento2/proyecto/modelos/detector_barbijo_20epochs.h5')
#Objetivo abrir la webcam  capturar la pantalla para eso usamos video capture de open cv
cap=cv.VideoCapture(0)

#print(f'cv.COLOR_RGD2GRAY = {cv.COLOR_RGB2GRAY}')
while True:
    ret, img = cap.read() # con esto leemos la webcam. 
    #ret es un boolen y img es (escencialmente) un np.array y es en rgd (matriz de tres dimensiones, alto ancho y cantidad de canales)
    ing = cv.flip( img, 1) #esto produce un flip horizontal, espejo. 1 es el eje, el eje 0 seria en el eje x
    
    #convertir el frame a escala degrises. 
    img_gris=cv.cvtColor(img, cv.COLOR_RGB2GRAY) #para que el algortmo funcione tenemos que pasarle una escala de 
    #grises. Podriamos pasarle solo un canal pero no es tan exacto. 
    ##
    #img_gris_chica = cv.resize(img_gris, (100, 100))
    #agregar_imagen(img, img_gris_chica, 500, 200) #estas dos lineas eran solo un ejemplo 
    ##
    
    # detectar Caras, le damos multiscale para que detecte caras en diferentes escalas
    caras = detector_caras.detectMultiScale(img_gris, 1.3, 5)
    ##detectar caras con respecto al eje x de tal manera de mantener un estado que se pueda asociar con cada cara
    
    #if len(caras) > 1:
    #    caras = caras[caras[0,:].argsort()]
    
    for i, (x,y,w,h) in enumerate(caras): 
        filtros[i].agregar_a_imagen(img, x, y, w, h)
        cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0))
    cv.imshow('Titulo de la ventana', img)
    k = cv.waitKey(30)
    if k == 27: #Esto quiere decir la tecla ESC
        break
cap.release()
cv.destroyAllWindows

