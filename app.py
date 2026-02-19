import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import io
from mtcnn import MTCNN
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'shamstabrez'

# Load Model
model_path = os.path.join("model", "RAFDB_Custom.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = tf.keras.models.load_model(model_path)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = (48, 48)
detector = MTCNN()

streaming = False

def detect_and_classify(frame):
    faces = detector.detect_faces(frame)
    detected_faces = []
    if faces:
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cropped_face = frame[y:y+h, x:x+w]

            if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                face_rgb = cv2.resize(cropped_face, IMG_SIZE)
                face_array = tf.keras.preprocessing.image.img_to_array(face_rgb) / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                predictions = model.predict(face_array)[0]
                top_indices = np.argsort(predictions)[-3:][::-1]  # Get top 3 emotions
                top_emotions = [(class_labels[i], round(predictions[i] * 100, 2)) for i in top_indices]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)  # Thin border
                for i, (emotion, percentage) in enumerate(top_emotions):
                    text = f"{emotion} ({percentage}%)"
                    cv2.putText(frame, text, (x, y - (i * 20) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                detected_faces.extend(top_emotions)
    return frame, detected_faces

@app.route('/')
def index():
    return render_template('index.html', top_emotions=None, img_base64=None, show_upload=True, show_camera=False, initial_image=True)

@app.route('/classify', methods=['POST'])
def classify_image():
    image = request.files['image']
    img = Image.open(image)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    processed_frame, top_emotions = detect_and_classify(img)

    _, buffer = cv2.imencode('.png', processed_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return render_template('index.html', top_emotions=top_emotions, img_base64=img_base64, show_upload=True, show_camera=False, initial_image=False)

@app.route('/predict_video', methods=['POST'])
def predict_video():
    data = request.json
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "No image data received"}), 400

    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    processed_frame, detected_faces = detect_and_classify(frame)
    _, buffer = cv2.imencode(".jpg", processed_frame)
    processed_image_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"processed_frame": processed_image_base64, "emotions": detected_faces})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
