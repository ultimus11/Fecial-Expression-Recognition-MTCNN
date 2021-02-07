from fer import FER
#from mod_emotions import U_MOD_img
import cv2

img = cv2.imread("imgjust1.jpg")
detector = FER()
#detector = U_MOD_img()
emot=detector.detect_emotions(img)
print(emot)
emotion,score=detector.top_emotion(img)
print(emotion)
print(score)