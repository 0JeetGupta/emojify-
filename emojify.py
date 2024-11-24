import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator

os.chdir("C:\\Users\\jeetg\\code\\emojify")

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights("C:\\Users\\jeetg\\code\\emojify\\emojify_model.weights.h5")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


emoji_dist={0:"emojis/angry.png",1:"emojis/disgusted.png",2:"emojis/fearful.png",3:"emojis/happy.png",4:"emojis/neutral.png",5:"emojis/sad.png",6:"emojis/surprised.png"}

class EmotionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1400x900+100+10")
        self.window['bg'] = 'black'

        # Load logo
        self.img = ImageTk.PhotoImage(Image.open("logo.png"))
        self.heading = Label(window, image=self.img, bg='black')
        self.heading.pack()

        # Title
        self.heading2 = Label(window, text="Photo to Emoji", pady=20, 
                               font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
        self.heading2.pack()

        # Camera feed label
        self.lmain = tk.Label(window, padx=50, bd=10)
        self.lmain.pack(side=LEFT)
        self.lmain.place(x=50, y=250)

        # Emoji label
        self.lmain2 = tk.Label(window, bd=10)
        self.lmain2.pack(side=RIGHT)
        self.lmain2.place(x=900, y=350)

        # Emotion text label
        self.lmain3 = tk.Label(window, bd=10, fg="#CDCDCD", bg='black')
        self.lmain3.pack()
        self.lmain3.place(x=960, y=250)

        # Exit button
        self.exitbutton = Button(window, text='Quit', fg="red", 
                                 command=window.destroy, 
                                 font=('arial', 25, 'bold'))
        self.exitbutton.pack(side=BOTTOM)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        
        # Haar cascade classifier
        self.bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Current emotion
        self.current_emotion = 0

        # Start video
        self.update()
        
        self.window.mainloop()

    def update(self):
        # Capture frame
        ret, frame = self.cap.read()
        
        if ret:
            # Resize frame
            frame = cv2.resize(frame, (600, 500))
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            num_faces = self.bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
            
            # Process each face
            for (x, y, w, h) in num_faces:
                # Draw rectangle
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                
                # Prepare ROI for emotion prediction
                roi_gray = gray_frame[y:y+h, x:x+w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                
                # Predict emotion
                prediction = emotion_model.predict(cropped_img)
                emotion_index = int(np.argmax(prediction))
                
                # Add text
                cv2.putText(frame, emotion_dict[emotion_index], 
                            (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Update current emotion
                self.current_emotion = emotion_index

            # Convert to RGB for Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update camera feed
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
            
            # Update emoji
            emoji_frame = cv2.imread(emoji_dist[self.current_emotion])
            emoji_rgb = cv2.cvtColor(emoji_frame, cv2.COLOR_BGR2RGB)
            emoji_img = Image.fromarray(emoji_rgb)
            emoji_imgtk = ImageTk.PhotoImage(image=emoji_img)
            
            self.lmain2.imgtk2 = emoji_imgtk
            self.lmain2.configure(image=emoji_imgtk)
            
            # Update emotion text
            self.lmain3.configure(text=emotion_dict[self.current_emotion], 
                                  font=('arial', 45, 'bold'))

        # Schedule next update
        self.window.after(10, self.update)

# Run the app
if __name__ == '__main__':
    root = tk.Tk()
    app = EmotionApp(root, "Photo to Emoji")