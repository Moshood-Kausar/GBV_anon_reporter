import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import av

# Load OpenCV face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class FaceBlurTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.GaussianBlur(face, (99, 99), 30)
            img[y:y+h, x:x+w] = face

        return img

st.title("🛡️ GBV Real-Time Face Blur Reporting System")

st.info("This app uses your webcam to record and blur your face in real-time for privacy.")

webrtc_streamer(key="blur", video_transformer_factory=FaceBlurTransformer)
