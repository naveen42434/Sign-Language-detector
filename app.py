import cv2
import streamlit as st
import tensorflow
import numpy as np
from tensorflow.keras import layers
from keras.models import load_model


def resize(img):
  image_1_resize = cv2.resize(img,(256,256))
  image_1_b_w = cv2.cvtColor(image_1_resize, cv2.COLOR_BGR2GRAY )
  return image_1_b_w

def absdiff(img1,img2):
  absdiff = cv2.absdiff(img1,img2)
  return absdiff

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
loaded_model=load_model(r"C:\Users\nk424\Downloads\DeepVisionModel.h5")
font=cv2.FONT_HERSHEY_COMPLEX

while run:
    _, frame = camera.read()
    img1 = resize(frame)
    #print(img1)
    cv2.waitKey(1)
    _, frame1 = camera.read()
    img2 = resize(frame1)
    #print(_, frame1)
    cv2.waitKey(1)
    abdiff=absdiff(img1,img2)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    ans = np.dstack([abdiff,abdiff,abdiff])
    ans = np.expand_dims(ans, axis = 0)
    
    eval=loaded_model.predict(ans)
    if eval==0:
      text="signed"
    else:
      text="unsigned"

    display=cv2.putText(frame,text,(50,50),font,1,(0,255,255),2,cv2.LINE_4)
    FRAME_WINDOW.image(display)
       
    #print(eval)
    #FRAME_WINDOW.image(abdiff)
else:
    st.write('Stopped')
