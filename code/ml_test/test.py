from mod_emotions import U_MOD_img
import cv2
import os

#original_file="imgjust1.jpg"
capture = cv2.VideoCapture(0)
ret, frame = capture.read()
if ret!=True:
	raise ValueError("cant read frame")
#os.remove(original_file)
cv2.imwrite('imgjust.jpg',frame)
cv2.imshow('img1', frame)
cv2.waitKey()


#here is the recognition algo
img = cv2.imread("imgjust.jpg")
#detector = FER()
detector = U_MOD_img()
emot=detector.detect_emotions(img)
print(emot)
emotion,score=detector.top_emotion(img)
print(emotion)
print(score)