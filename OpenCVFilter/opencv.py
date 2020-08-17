import cv2 as cv 
import imageio
import os #util para manipular el sistema operativo 
#Objetivo abrir la webcam  capturar la pantalla para eso usamos video capture de open cv
cap=cv.VideoCapture(0)
while True:
    ret, img = cap.read() # con esto leemos la webcam. 
    #ret es un boolen y img es (escencialmente) un np.array y es en rgd (matriz de tres dimensiones, alto ancho y cantidad de canales)
    ing = cv.flip( img, 1) #esto produce un flip horizontal, espejo. 1 es el eje, el eje 0 seria en el eje x
    cv.imshow('Titulo', img) #con esto imprimimos cada captura de open cv en el while
    #open cv también nos permite hacer llamadas a eventos 
    k = cv.waitKey(30)
    if k == 27: #Esto quiere decir la tecla ESC
        break
cap.release()
cv.destroyAllWindows

