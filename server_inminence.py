from alga import PictureInput, JsonOutput, Json
import cv2
import face_recognition
import numpy as np
import os
import time
from keras.models import load_model


# #Implementar BD interna del veh
from alga import PictureInput, JsonOutput, Json

start_total = time.time()

def segment(img):
    return img

def state_recognition(img, name_prove, classNames):
    
    im = ((img.data+1)*255/2).astype('uint8') #Para formatear el tipo del frame de la imagen que pasa del cliente al servidor

    faceDistances = []
    authorizeEmbeddings = []
    permission = ""

    for filename in os.listdir("../Images"):
        authorizeImages = cv2.imread(os.path.join("../Images",filename))
        classNames.append(filename)
        if authorizeImages is not None:
            authorizeEmbeddings.extend(face_recognition.face_encodings(authorizeImages))


    facesCurrFrame = face_recognition.face_locations(im)
    encodesCurrFrame = face_recognition.face_encodings(im, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(authorizeEmbeddings, encodeFace)
        faceDistance = face_recognition.face_distance(authorizeEmbeddings, encodeFace)
        ##Mientras sea menor o igual que 0.6 se tratará de la misma persona.
        faceDistances.append(faceDistance)
        matchIndex = np.argmin(faceDistance)
        print(faceDistances)
        print(matches)

        if matches[matchIndex]:
            name_prove = classNames[matchIndex]
            name = classNames[matchIndex].replace(".jpg", "").replace("_", " ")
            print(name)
        else:
            print("Unknown.")

        if matches[matchIndex]:
            permission = "Authorized."
        else:
            permission = "Refused."

    return permission, name_prove, classNames

def state_drowsiness(img, score):

        start_drow = time.time()

        # Establecer el clasificador Haar cascade para la cara y los ojos
        face = cv2.CascadeClassifier('Drowsiness detection/haarCascadeFiles/haarcascade_frontalface_alt.xml')
        leye = cv2.CascadeClassifier('Drowsiness detection/haarCascadeFiles/haarcascade_lefteye_2splits.xml')
        reye = cv2.CascadeClassifier('Drowsiness detection/haarCascadeFiles/haarcascade_righteye_2splits.xml')

        model = load_model('Drowsiness detection/models/cnnCat2.h5') # Se carga el modelo para predecir cada ojo.

        im=((img.data+1)*255/2).astype('uint8')
                
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        rpred=[99]
        lpred=[99]
        # Variable para ver el estado de los ojos (abiertos/cerrados)
        state = ""

        # Detección de la cara. Devuelve un array de detecciones con coordenadas x,y, la altura y el ancho de la caja límite del objeto. 

        # Detección del ojo derecho.
        for (x,y,w,h) in right_eye:
            r_eye=im[y:y+h,x:x+w] # Para extraer solo los datos de los ojos de la imagen completa se extrae la caja de límites del ojo y luego se saca la imagen del ojo del marco con este código.
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY) # Se convierte el color de la imagen a escala de grises.
            r_eye = cv2.resize(r_eye,(24,24)) # Se cambia el tamaño de la imagen a 24*24 píxeles, ya que el modelo fue entrenado en imágenes de 24*24 píxeles.
            r_eye= r_eye/255 # Se normalizan los datos para una mejor convergencia (todos los valores estarán entre 0 y 1)
            r_eye=  r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict(r_eye) # Si el valor de rpred[0] = 1, indica que los ojos están abiertos, si el valor de rpred[0] = 0 entonces, indica que los ojos están cerrados.
            rpred = np.argmax(rpred, axis=1)
            break

        # Detección del ojo izquierdo.
        for (x,y,w,h) in left_eye:
            l_eye=im[y:y+h,x:x+w] 
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict(l_eye)
            lpred = np.argmax(lpred, axis=1)
            break

        if ((rpred[0]==0) & (lpred[0]==0)): # Si los dos ojos están cerrados, se aumenta la puntuación y cuando los ojos estén abiertos, se disminuye la puntuación.
            score=score+1
        else:
            score=score-1
        
        if(score<0):
            score=0
        if(score>30): # Si la puntuación es superior a 10, significa que los ojos de la persona están cerrados durante un largo periodo de tiempo. Es entonces cuando suena la alarma.
            state = "Dormido"
            score = 30
        else:
            state = "Despierto"

        end_drow = time.time()
        print("\nEl tiempo de la función drowsiness es de: ", end_drow - start_drow, "\n\n")

        return state,score

def run_segmentation_test(inaddr,json_outaddr, slice='/', hwm = 1, multithreading=False):
    with PictureInput(inaddr, f'{slice}video/', hwm = hwm, multithreading=multithreading) as input, \
            JsonOutput(json_outaddr, f'{slice}meta/', bind=True, hwm = hwm, multithreading=multithreading) as output_json:

        metadata = Json()
        classNames = []
        name_prove = ''
        score = 0

        cont = 50

        while cont >= 1:
            start_rec = time.time()


            img = input.recv()

            if not img is None:
                permission = state_recognition(img, name_prove, classNames)
                if (permission[0] == "Authorized."): 
                    # Send metadata.
                    metadata.data = "Authorized."
                    output_json.send(metadata)
                else: #(permission[0] == "Refused."):
                    metadata.data = "Refused."
                    output_json.send(metadata)
                name_prove = permission[1]
                classNames = permission[2]
                end_rec = time.time()
                cont = cont - 1
                print("\nEl tiempo del reconocimiento es de: ", end_rec - start_rec, "\n\n")

        while True:
            if not img is None:

                stat = state_drowsiness(img, score)
                #print("Estado recibido: ", stat[0])
                if (stat[0] == "Despierto"):
                    metadata.data = "Despierto"
                    output_json.send(metadata)
                elif (stat[0] == "Dormido"):
                    metadata.data = "Dormido"
                    output_json.send(metadata)
                score = stat[1]



if __name__ == '__main__':

    inaddr = 'tcp://*:8000'
    json_outaddr = 'tcp://*:8060'

    slice = '/dr/'
    hwm = 1
    multithreading = True
    name = "user_0"

    run_segmentation_test(inaddr,json_outaddr, slice, hwm, multithreading)


end_total = time.time()

print("\nEl tiempo total de acceso al servidor es de: ", end_total - start_total, "\n\n")