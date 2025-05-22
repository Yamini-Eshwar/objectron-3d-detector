import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image 
import base64
import matplotlib.pyplot as plt
import io
import tempfile
import time


# ==== Set background image using HTML + CSS ====
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()  # Correct way to encode
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# === Use your local image here ===
set_bg("C:/Users/vamsi/Downloads/col.jpg")  # update path if needed

# side bar select- mode
mode = st.sidebar.selectbox("Choose Detection Mode", ["Video Detection", "Image Detection"])

st.title("MediaPipe Objectron - 3D Object Detection")

if mode == "Video Detection":
    st.write("Video Detection Mode Selected")
   
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        mp_objectron = mp.solutions.objectron
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(tfile.name)

        model_type = st.sidebar.selectbox('Select Object Type', ["Chair", "Camera", "Cup", "Shoe"])
        objectron = mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.3, min_tracking_confidence=0.7, model_name='Shoe')

        stframe = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = objectron.process(image)

            if results.detected_objects:
                for obj in results.detected_objects:
                    mp_drawing.draw_landmarks(image, obj.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(image, obj.rotation, obj.translation)

            stframe.image(image, channels='RGB', use_column_width=True)
            time.sleep(0.03)

        cap.release()
        st.success("Video processing complete.")


elif mode == "Image Detection":
    st.write("Image Detection Mode Selected")

    mp_objectron = mp.solutions.objectron
    mp_drawing = mp.solutions.drawing_utils

    image_files = {
    "Chair": r"C:\Users\vamsi\Downloads\Electricchair.jpg",
    "Camera": r"C:\Users\vamsi\Downloads\camera.jpg",
    "Cup": r"C:\Users\vamsi\Downloads\mycupp.jpg",
    "Shoe": r"C:\Users\vamsi\Downloads\close.jpg"
    }

    selected_image_name = st.sidebar.selectbox("Choose Image", list(image_files.keys()))
    image_path = image_files[selected_image_name]

    image = cv2.imread(image_path)

    mug = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # step 2: LOAD THE OBJECT DETECTION MODEL
    objectron = mp_objectron.Objectron(static_image_mode=True, max_num_objects=5, min_detection_confidence=0.2, model_name= selected_image_name)

    # step 3: RUN THE DETECTION 
    results = objectron.process(mug)
    # step 4: CHECK IF OBJECT IS FOUND
    if results.detected_objects:
       # step 5: DRAW THE OBJECT ON IMAGE
        annotated_image = mug.copy()  # copy of the original image

        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS )
            
        image_pil = Image.fromarray(annotated_image)
        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        encoded_image = base64.b64encode(buf.getvalue()).decode()

        st.markdown(f"""<div style="text-align:center;"><img src="data:image/png;base64,{encoded_image}" style="border-radius: 20px;
          box-shadow: 0 0 40px 10px rgba(0, 255, 0, 0.6);
                    max-width: 100%; height: auto;" />
        <p style="color: white; font-size: 20px;">3D Detection Result</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        print('No box landmarks detected.')



