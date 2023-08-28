
import cv2
import os
import imutils
canNombre="zar"
dataPath="C:/Users/ALBERTO1/Desktop/Imagenes/Data"  #Ubicación de la carpeta donde se encuentra la Data#
canPath  = dataPath + '/' + canNombre
if not os.path.exists(canPath):
	print('Carpeta creada: ',canPath)
	os.makedirs(canPath)

#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture('C:/Users/ALBERTO1/Desktop/Imagenes/zar.mp4') #Ubicación donde se encuentra el video que utilizaremos como ejemplo#

faceClassif = cv2.CascadeClassifier('mydog.xml')
count = 0

while (cap.isOpened()): #Seccion de código que detecta y guarda las imagenes detectadas #

	ret, frame = cap.read()
	if ret == False: break
	frame =  imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = frame.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5,75)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(50,50),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(canPath + '/rostro_{}.jpg'.format(count),rostro)
		count = count + 1
	cv2.imshow('frame',frame)

	k =  cv2.waitKey(1)
	if k == 27 or count >= 300:
		break

cap.release()
cv2.destroyAllWindows()
