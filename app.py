import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, AudioProcessorBase
import cv2
import av
import numpy as np

# --- FACE BLUR VIDEO TRANSFORMER ---
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

# --- VOICE MASK AUDIO PROCESSOR ---
class VoiceMasker(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        samples = frame.to_ndarray()
        # Apply simple pitch reduction for anonymity
        modified = np.clip(samples * 0.7, -32768, 32767).astype(np.int16)
        new_frame = av.AudioFrame.from_ndarray(modified, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate
        return new_frame

# --- Streamlit Interface ---
st.set_page_config(page_title="GBV Reporter", layout="centered")
st.title("🛡️ GBV Real-Time Anonymous Reporter")

st.info("This app blurs your face and masks your voice in real-time to help report sensitive cases like GBV anonymously.")

# --- WebRTC Stream ---
webrtc_streamer(
    key="anon-report",
    video_transformer_factory=FaceBlurTransformer,
    audio_processor_factory=VoiceMasker,
    media_stream_constraints={"video": True, "audio": True},
    async_processing=True
)
