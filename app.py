import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image

from streamlit_webrtc import (
    webrtc_streamer,
    VideoTransformerBase
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "deep_cnn_model.keras"
    )

model = load_model()

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# --------------------------------------------------
# FACE DETECTOR
# --------------------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_emotion(face_gray):

    face_gray = cv2.resize(
        face_gray,
        (48, 48)
    )

    face_input = np.expand_dims(
        face_gray,
        axis=(0, -1)
    ).astype("float32")

    prediction = model.predict(
        face_input,
        verbose=0
    )

    emotion_index = np.argmax(
        prediction
    )

    emotion = class_names[
        emotion_index
    ]

    confidence = (
        np.max(prediction) * 100
    )

    return (
        emotion,
        confidence,
        prediction[0]
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(
    "😊 Face Emotion Recognition"
)

st.markdown(
    "Upload an image or use your webcam for live emotion detection."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

mode = st.sidebar.radio(
    "Choose Mode",
    [
        "Upload Image",
        "Live Camera"
    ]
)

# ==================================================
# IMAGE MODE
# ==================================================

if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )

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

                emotion, confidence, probs = (
                    predict_emotion(face)
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{emotion} {confidence:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            st.image(
                frame,
                use_container_width=True
            )

            st.success(
                f"Detected Emotion: {emotion}"
            )

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

            st.subheader(
                "Emotion Probabilities"
            )

            prob_dict = {}

            for i in range(
                len(class_names)
            ):

                prob_dict[
                    class_names[i]
                ] = float(
                    probs[i] * 100
                )

            st.bar_chart(
                prob_dict
            )

# ==================================================
# LIVE CAMERA MODE
# ==================================================

if mode == "Live Camera":

    st.subheader(
        "Live Emotion Detection"
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

                emotion, confidence, _ = (
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
        key="emotion-detection",
        video_transformer_factory=
        EmotionDetector
    )