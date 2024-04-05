from flask import Flask, request, send_file, jsonify, render_template, make_response
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from io import BytesIO
import cv2
import numpy as np

app = Flask(__name__)

# Load the models
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classifier = load_model("model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    # Serving the frontend HTML (assuming it's named 'index.html' in a 'templates' folder)
    return render_template('index.html')

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    file = request.files['image']
    if not file:
        return jsonify({'error': 'no file uploaded'}), 400

    frame = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.png', frame)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    return response



    # return send_file(response, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',ssl_context=('cert.pem', 'key.pem'))
