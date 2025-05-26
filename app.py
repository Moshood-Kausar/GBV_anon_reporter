import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import av

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define video transformer
class FaceBlurTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Apply Gaussian blur to detected faces
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.GaussianBlur(face, (99, 99), 30)
            img[y:y+h, x:x+w] = face

        return img

# Streamlit app UI
st.title("GBV Anonymous Reporter")
st.markdown("📹 **Real-time face blurring for anonymous reporting**")

# WebRTC streamer with optimized client settings
webrtc_streamer(
    key="face-blur",
    video_transformer_factory=FaceBlurTransformer,
    client_settings=ClientSettings(
        media_stream_constraints={
            "video": {
                "width": {"ideal": 320},
                "height": {"ideal": 240},
                "frameRate": {"ideal": 15}
            },
            "audio": False  # Turn on later when voice anonymization is added
        },
        rtc_configuration={  # Optional: STUN server for better connection
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
)
