import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import tkinter
import random

class arayuz:
    
    def __init__(self, window,predicted_emotion):
         # create a button to call a function called 'say_hi'
        if predicted_emotion =='angry':
            self.text_btn = tkinter.Button(window, text = "Sinirli gözüküyorsun tavsiye ister misin?", command = self.say_hi)
            self.text_btn.pack()
        if predicted_emotion =='disgust':
            self.text_btn = tkinter.Button(window, text = "Tiksinmiş gözüküyorsun tavsiye ister misin?", command = self.say_hi)
            self.text_btn.pack()
        if predicted_emotion =='fear':
            self.text_btn = tkinter.Button(window, text = "Korkmuş gözüküyorsun tavsiye ister misin?", command = self.say_hi)
            self.text_btn.pack()
        if predicted_emotion =='happy':
            self.text_btn = tkinter.Button(window, text = "Mutlu gözüküyorsun tavsiye ister misin?", command = self.say_hi)
            self.text_btn.pack()
        if predicted_emotion =='sad':
            self.text_btn = tkinter.Button(window, text = "Üzgün gözüküyorsun tavsiye ister misin?", command = self.say_hi)
            self.text_btn.pack()
        if predicted_emotion =='surprise':
            self.text_btn = tkinter.Button(window, text = "Şaşkın gözüküyorsun tavsiye ister misin?", command = self.say_hi)
            self.text_btn.pack()
            
            
        self.close_btn = tkinter.Button(window, text = "Close", command = window.destroy)
        self.close_btn.pack()

    def say_hi(self):
        angryS=["Derin derin nefes al ver. Bunu bir kaç kez yap.", 
                "Ayağa kalk biraz yürü ya da olduğun yerde biraz hareket et.", 
                "En rahatladığın o müziği aç ve gevşemene bak.",
                "Komik videolar izle.",
                "Aklından geçen şey bir kötülükse biraz ertele daha sonra tekrar düşünürsün."]
        disgustS=["Bir bardak soğuk su içmek en iyisi",
                  "İğrenmen geçene kadar başka bir şeye yönlen"]
        fearS=["Sen bunun üstesinden de gelebilirsin. Korkma.",
               "Sayfayı kapat korkun geçene kadar."]
        happyS=["Kendini bir tatlıyla ödüllendirmek iyi olacak.",  
                "Güzel bi şarkı açıp dans etmek şuan tam zamanı.",
                "Mutluluğunu paylaşmak harika olur. Birine telefon aç.",
                "Birilerine yardım et mutluluğu yaymak harika olur."]
        sadS=["Acıkmış olabilirsin mutluluğun ilk adımı tok olmak.",
              "Üzüntünü paylaşmak iyi gelebilir. Birini arayabilirsin."]
        supriseS=["Şimdi ne yapacağını düşün.",
                  "Ne düşündüğünü merak ettim biriyle bunu tartışmalısın."]
        if predicted_emotion =='angry':
            tkinter.Label(window, text = random.choice(angryS)).pack()            
        if predicted_emotion =='fear':
            tkinter.Label(window, text = random.choice(fearS)).pack()
        if predicted_emotion =='happy':
            tkinter.Label(window, text = random.choice(happyS)).pack()
        if predicted_emotion =='sad':
            tkinter.Label(window, text = random.choice(sadS)).pack()
        if predicted_emotion =='surprise':
            tkinter.Label(window, text = random.choice(supriseS)).pack()
        if predicted_emotion =='disgust':
            tkinter.Label(window, text = random.choice(disgustS)).pack()    
            
def location(window):
    window.title("GUI")
    w = 350 # width for the Tk root
    h = 150 # height for the Tk root

    ws = window.winfo_screenwidth() # width of the screen
    hs = window.winfo_screenheight() # height of the screen

    x = (ws/1.10) - (w/2)
    y = (hs/1.10) - (h/1.10)

    window.geometry('%dx%d+%d+%d' % (w, h, x, y))

    window.mainloop() # starts the mainloop

model = model_from_json(open("face_model.json", "r").read())  
model.load_weights('face_model.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0) 
h=0
a=0
s=0 
su=0
d=0
f=0
while True:  
    ret,test_img=cap.read()     
    if not ret:  
        continue  
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.35, 4)  
    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)  
        roi_gray=gray_img[y:y+w,x:x+h]  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img_pixels = image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
        predictions = model.predict(img_pixels)    
        max_index = np.argmax(predictions[0])  
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
        predicted_emotion = emotions[max_index]  
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if predicted_emotion == 'disgust' and d<3:
            d=d+1
            window = tkinter.Tk()
            arayuz(window,predicted_emotion)
            location(window)
        if predicted_emotion == 'fear' and f<3:
            f=f+1
            window = tkinter.Tk()
            arayuz(window,predicted_emotion)
            location(window)
        if predicted_emotion == 'surprise' and su<3:
            su=su+1
            window = tkinter.Tk()
            arayuz(window,predicted_emotion)
            location(window)
        if predicted_emotion == 'angry' and a<3:
            a=a+1
            window = tkinter.Tk()
            arayuz(window,predicted_emotion)
            location(window)
        if predicted_emotion == 'happy' and h<3:
            h=h+1
            window = tkinter.Tk()
            arayuz(window,predicted_emotion)
            location(window)
        if predicted_emotion == 'sad' and s<3:
            s=s+1
            window = tkinter.Tk()
            arayuz(window,predicted_emotion)
            location(window)
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)

    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows