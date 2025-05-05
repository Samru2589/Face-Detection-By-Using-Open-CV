# to detect object of interest(face) in real time and to keep tracking of the same object.This is a simple example of how to detect face in Python. You can try to use training samples of any other object of your choice to be detected by training the classifier on required objects.

# OpenCV program to detect face in real time
# import libraries of python OpenCV 
# where its functionality resides
import cv2 

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier(r"C:\Users\Samruddhi Yadav\Documents\Resume Project\Open CV\haarcascade_frontalface_default.xml")

# https://github.com/Itseez/opencv/blob/master
# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
eye_cascade = cv2.CascadeClassifier(r"C:\Users\Samruddhi Yadav\Documents\Resume Project\Open CV\haarcascade_eye.xml") 

# capture frames from a camera
cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized.
while 1: 

    # reads frames from a camera
    ret, img = cap.read() 

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray) 

        #To draw a rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

    # Display an image in a window
    cv2.imshow('img',img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()


import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit UI
st.title("👤 Face Detection using OpenCV & Streamlit")
st.write("Upload an image or use webcam to detect faces.")

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

# --- Option 1: Upload Image ---
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(r"C:\Users\Samruddhi Yadav\Documents\Documents\Photo (2) - Copy.jpeg")
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

        st.image(img_array, caption=f"Detected {len(faces)} face(s).", use_column_width=True)

# --- Option 2: Webcam ---
elif option == "Use Webcam":
    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])

    cap = None
    if run:
        cap = cv2.VideoCapture(0)

    while run and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if cap:
        cap.release()
