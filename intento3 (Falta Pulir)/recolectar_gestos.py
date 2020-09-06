#python intento3/recolectar_gestos.py --nombre fondo --dir C:\Users\carlo\Desktop\GitHub\FiltersForSelfies\intento3\capturas
#la primera linea es para llamar al programa darle el nombre fondo a la carpeta y los archivos que se guaradar y la ubicacion. 
import cv2 as cv
import os
import numpy as np
import argparse 

script_dir = os.path.dirname(os.path.realpath(__file__))
image_dir = os.path.join(script_dir, 'entrenamiento')



aparser = argparse.ArgumentParser()
required_arguments = aparser.add_argument_group('required arguments')
required_arguments.add_argument('--nombre',
                                help='nombre de la clase',
                                required=True)
required_arguments.add_argument('--dir',
                                help='directorio donde guardar las imagenes',
                                required=True)
required_arguments.add_argument('--cant',
                                help='cantidad de imagenes',
                                type=int,
                                default=200)
required_arguments.add_argument('--dimension',
                                help='cantidad de imagenes',
                                type=int,
                                default=300)
required_arguments.add_argument('--dimension-salida',
                                help='cantidad de imagenes',
                                type=int,
                                default=200)
args = aparser.parse_args()


detector_caras = cv.CascadeClassifier('../FiltersForSelfies/OpenCVFilter/haarcascade_frontalface_default.xml')

# capturar imagen desde la webcam
cap = cv.VideoCapture(0)

CAPTURAR = False
imagenes_generadas = 0

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
        
        if CAPTURAR and imagenes_generadas <= args.cant:
            ejemplo = img[y:y+h, x:x+w] #recortar el rectangulo que nos interesa
            isDir=os.path.isdir(os.path.join(args.dir, args.nombre))
            if not isDir:
                os.mkdir(os.path.join(args.dir, args.nombre), 755)
            filename = os.path.join(args.dir, args.nombre, f'{args.nombre}_{imagenes_generadas}.jpg')
            cv.imwrite(filename, ejemplo)
            imagenes_generadas += 1
            print(f'generadas {imagenes_generadas} imagenes')
            
        cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255))
    
    
    
    cv.imshow('Ttulo de la ventana', img)
    k = cv.waitKey(30)
    if k == 27: # ESC (ASCII)
        break

    elif k == ord('f'):
        CAPTURAR = ~CAPTURAR
        
cap.release()
cv.destroyAllWindows()