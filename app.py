import cv2
import streamlit as st
import numpy as np
from keras.models import load_model
import tempfile


def resize(img):
    image_1_resize = cv2.resize(img, (256, 256))
    image_1_b_w = cv2.cvtColor(image_1_resize, cv2.COLOR_BGR2GRAY)
    return image_1_b_w


def absdiff(img1, img2):
    absdiff = cv2.absdiff(img1, img2)
    return absdiff


st.title("Video File Processing")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Upload video file
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

# Load the model
loaded_model = load_model(r"DeepVisionModel.h5")
font = cv2.FONT_HERSHEY_COMPLEX

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if run:
        camera = cv2.VideoCapture(tfile.name)
        if not camera.isOpened():
            st.error("Error: Could not open video file.")
        else:
            while run:
                ret, frame = camera.read()
                if not ret:
                    st.write("Video processing completed.")
                    break
                img1 = resize(frame)
                cv2.waitKey(1)
                ret, frame1 = camera.read()
                if not ret:
                    st.write("Video processing completed.")
                    break
                img2 = resize(frame1)
                cv2.waitKey(1)
                abdiff = absdiff(img1, img2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

                ans = np.dstack([abdiff, abdiff, abdiff])
                ans = np.expand_dims(ans, axis=0)

                eval = loaded_model.predict(ans)
                if eval == 0:
                    text = "signed"
                else:
                    text = "unsigned"

                display = cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
                FRAME_WINDOW.image(display)

            camera.release()
else:
    st.write('Stopped')