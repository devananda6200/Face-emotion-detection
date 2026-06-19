import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.big-font {
    font-size:28px !important;
    font-weight:bold;
    color:#00D4FF;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "deep_cnn_model.keras",
        compile=False
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# ==================================================
# CLASS NAMES
# ==================================================

class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# ==================================================
# FACE DETECTOR
# ==================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_emotion(face):

    # Ensure grayscale
    if len(face.shape) == 3:
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2GRAY
        )

    # Resize exactly like training
    face = cv2.resize(
        face,
        (48, 48)
    )

    face = face.astype(np.float32)

    face_input = np.expand_dims(
        face,
        axis=(0, -1)
    )

    prediction = model.predict(
        face_input,
        verbose=0
    )

    emotion_index = np.argmax(
        prediction[0]
    )

    confidence = (
        np.max(prediction[0]) * 100
    )

    emotion = class_names[
        emotion_index
    ]

    return (
        emotion,
        confidence,
        prediction[0],
        face
    )

# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<p class="big-font">😊 Face Emotion Recognition</p>',
    unsafe_allow_html=True
)

st.write(
    "Detect emotions from uploaded images or live webcam."
)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Options")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Upload Image",
        "Live Camera"
    ]
)

st.sidebar.info(
    """
    Supported emotions:
    
    • Angry  
    • Disgust  
    • Fear  
    • Happy  
    • Neutral  
    • Sad  
    • Surprise
    """
)

# ==================================================
# IMAGE UPLOAD MODE
# ==================================================

if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        frame = np.array(image)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:

            st.error(
                "No face detected."
            )

        else:

            result_col1, result_col2 = st.columns(2)

            for (
                x,
                y,
                w,
                h
            ) in faces:

                face = gray[
                    y:y+h,
                    x:x+w
                ]

                (
                    emotion,
                    confidence,
                    probs,
                    processed_face
                ) = predict_emotion(face)

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{emotion} ({confidence:.1f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            with result_col1:

                st.image(
                    frame,
                    caption="Detected Faces",
                    use_container_width=True
                )

            with result_col2:

                st.image(
                    processed_face,
                    caption="Face Sent To Model",
                    width=250
                )

                st.success(
                    f"Emotion: {emotion}"
                )

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )

            st.subheader(
                "Emotion Probabilities"
            )

            prob_dict = {
                class_names[i]:
                float(probs[i] * 100)
                for i in range(
                    len(class_names)
                )
            }

            st.bar_chart(
                prob_dict
            )

# ==================================================
# LIVE CAMERA MODE
# ==================================================

if mode == "Live Camera":

    st.subheader(
        "📷 Live Emotion Detection"
    )

    class EmotionDetector(
        VideoTransformerBase
    ):

        def transform(
            self,
            frame
        ):

            img = frame.to_ndarray(
                format="bgr24"
            )

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (
                x,
                y,
                w,
                h
            ) in faces:

                face = gray[
                    y:y+h,
                    x:x+w
                ]

                emotion, confidence, _, _ = (
                    predict_emotion(face)
                )

                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    f"{emotion} {confidence:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            return img

    webrtc_streamer(
        key="emotion-camera",
        video_transformer_factory=
        EmotionDetector
    )