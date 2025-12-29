import cv2
import os
import numpy as np
import time

#Intevalo entre capturas
intervalo=0.5
ultimo_tempo=0

#Váriavel de para começar capturas
play=False

# Caminho Haarcascade
cascPath = 'cascade/haarcascade_frontalface_default.xml'
cascPathOlho = 'cascade/haarcascade-eye.xml'

# Classifier baseado nos haarcascade
facePath = cv2.CascadeClassifier(cascPath)
facePathOlho = cv2.CascadeClassifier(cascPathOlho)
video_capture = cv2.VideoCapture(0)


increment = 1    #contador de capturas
numMostras = 40 #delimitador de capturas
id = input('Digite seu identificador: ') # indentificador da face capturada
person_id=input('Digite seu nome: ')
width, height = 220, 220 #tamanho da imagem capturada
print('Capturando as faces...')

# Create directory para salvar on images
os.makedirs('fotos',exist_ok=True)


while (True):
    key = cv2.waitKey(1) & 0xFF


    conect, image = video_capture.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Qualidade da luz sobre a imagem capturada
    print(np.average(gray))

    # Realizando face detect
    face_detect = facePath.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minSize=(35, 35),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in face_detect:
        # Desenhando retangulo na face detectada
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Realizando deteccao do olho da face
        region = image[y:y + h, x:x + w]
        imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        face_detect_olho = facePathOlho.detectMultiScale(imageOlhoGray)
        for (ox, oy, ow, oh) in face_detect_olho:
            # Desenhando retangulo nos olhos detectados
            cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)

        # inicia a captura apenas no clique da tecla
        if key == ord('j'):
            play = True
            print('Modo captura iniciado...')

        if play:
            #garante o intervalo entre capturas
            tempo_atual = time.time()
            if tempo_atual - ultimo_tempo > intervalo:
                #define a captura e a salva no diretório ('fotos')
                face_off = cv2.resize(gray[y:y + h, x:x + w], (width, height))
                cv2.imwrite('fotos/'+person_id+'.' + str(id) + '.' + str(increment) + '.jpg',face_off)

                print('[Foto ' + str(increment) + ' capturada com sucesso] - ', np.average(gray))

                increment += 1
                ultimo_tempo=tempo_atual

    cv2.imshow('Face', image)
    cv2.waitKey(1)



    if increment > numMostras: break

print('Fotos capturadas com sucesso :)')
video_capture.release()
cv2.destroyAllWindows()