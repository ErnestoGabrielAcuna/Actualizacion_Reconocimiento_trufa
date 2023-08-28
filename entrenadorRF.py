from builtins import print

import cv2
import os
import numpy as np
dataPath='C:/Users/ALBERTO1/Desktop/Imagenes/Data'
canList=os.listdir(dataPath)
print("Lista de canes: ", canList)
labels=[]
facesData=[]
label=0
for nameDir in canList:
    canPath=dataPath + '/' + nameDir
    print("Leyendo las imagenes")

    for fileName in os.listdir(canPath):
        print("Rostros: " + nameDir +"/"+ fileName )
        labels.append(label)
        facesData.append(cv2.imread(canPath+"/"+fileName,0))
        #image=cv2.imread(canPath+"/"+fileName,0)
        #cv2.imshow("image", image)
        #cv2.waitKey(10)
        label=label+1
        #print("labels=", label)
# Entrenando el reconocedor de rostros
print("Entrenando...")
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")



