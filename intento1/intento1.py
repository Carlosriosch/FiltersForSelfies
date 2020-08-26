import cv2 as cv 
import imageio 
from add_mask1 import agregar_imagen
from class_filtro1 import Filtro 
cap=cv.VideoCapture(0)
ret, img = cap.read()

#mascaras
mascara0=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/marte.png')
mascara1=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/astronave.png')
mascara2=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/monstruo.png')
mascara3=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/ovni.png')
mascara4=imageio.imread('../FiltersForSelfies/OpenCVFilter/Icons/mercurio.png')
Filtros1=[Filtro(mascara0), Filtro(mascara1), Filtro(mascara2), Filtro(mascara3), Filtro(mascara4)]


detector_caras = cv.CascadeClassifier('../FiltersForSelfies/OpenCVFilter/haarcascade_frontalface_default.xml')
while True:
    ret, img = cap.read() # con esto leemos la webcam. 
    #ret es un boolen y img es (escencialmente) un np.array y es en rgd (matriz de tres dimensiones, alto ancho y cantidad de canales)
    ing = cv.flip( img, 1)
    img_gris=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    caras = detector_caras.detectMultiScale(img_gris, 1.3, 5)

    for i, (x,y,w,h) in enumerate(caras): 
        ##TODO arreglar las coordenadas de las máscaras
        
        cv.rectangle(img, (x,y), (x+w,y+h), (200,55,32))
        Filtros1[0].agregador_imagen(img, x, y, x+122, y)
        Filtros1[1].agregador_imagen(img, x, y, x, y)
        Filtros1[2].agregador_imagen(img, x, y, x+200, y+150)
        Filtros1[3].agregador_imagen(img, x, y, x+122, y)
        Filtros1[4].agregador_imagen(img, x, y, x+122, y+122)


    cv.imshow('Titulo de la ventana', img)
    k = cv.waitKey(30)
    if k == 27: #Esto quiere decir la tecla ESC
        break
cap.release()
cv.destroyAllWindows

