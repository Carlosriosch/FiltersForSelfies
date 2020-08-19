import cv2 as cv 
import imageio
import os #util para manipular el sistema operativo 
from func_add_mask import agregar_imagen

mascara=imageio.imread('../FiltersForSelfies/OpenCVFilter/vinito.png')

##Obs podriamos usar YOLO pero es una herramienta poderosa de más. Mejor vamos a usar una herramienta de OpenCv llamada
#HAAR Cascade Classifier que sirve solo para detectar rostros

detector_cara = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#el problema es que este detector funciona solo para un canal, entonces tenemos que unir los canales (preprocesarlos)

#Objetivo abrir la webcam  capturar la pantalla para eso usamos video capture de open cv
cap=cv.VideoCapture(0)

print(f'cv.COLOR_RGD2GRAY = {cv.COLOR_RGB2GRAY}')
while True:
    ret, img = cap.read() # con esto leemos la webcam. 
    #ret es un boolen y img es (escencialmente) un np.array y es en rgd (matriz de tres dimensiones, alto ancho y cantidad de canales)
    ing = cv.flip( img, 1) #esto produce un flip horizontal, espejo. 1 es el eje, el eje 0 seria en el eje x
    
    #convertir el frame a escala degrises. 
    img_gris=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_gris_chica = cv.resize(img_gris, (100, 100))
    agregar_imagen(img, mascara, 200, 200)
    agregar_imagen(img, img_gris_chica, 500, 200)

    cv.imshow('Titulo', img) #con esto imprimimos cada captura de open cv en el while
    #open cv también nos permite hacer llamadas a eventos 
    k = cv.waitKey(30)
    if k == 27: #Esto quiere decir la tecla ESC
        break
cap.release()
cv.destroyAllWindows

