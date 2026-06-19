# Face Emotion Recognition

A simple Streamlit app for face emotion recognition using a pretrained Keras model and OpenCV face detection.

## Project Structure

- `app.py` - Main Streamlit application.
- `deep_cnn_model.keras` - Trained Keras model for emotion classification.
- `haarcascade_frontalface_default.xml` - OpenCV Haar cascade for face detection.
- `requirements.txt` - Python dependencies.

## Features

- Upload an image and detect face emotion.
- Live camera mode for real-time emotion detection.
- Displays emotion label, confidence, and probability chart.

## Requirements

- Python 3.8+ recommended
- `pip` package manager

## Install

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

From the project directory:

```bash
streamlit run app.py
```

Then open the displayed local URL in your browser.

## Usage

- Select **Upload Image** to upload a photo and detect emotion on detected faces.
- Select **Live Camera** to use your webcam for real-time emotion recognition.

## Notes

- The app uses a Haar cascade classifier for face detection.
- The emotion classes are: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
- If no face is detected in the uploaded image, the app displays an error.
