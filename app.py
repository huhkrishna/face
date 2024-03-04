from flask import Flask, render_template, request, jsonify
import cv2
import face_recognition
import numpy as np
import os

app = Flask(__name__)

# Load known faces and their labels
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            label = os.path.splitext(filename)[0]
            image_path = os.path.join(folder_path, filename)
            known_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(known_image)
            if face_encoding:
                known_face_encodings.append(face_encoding[0])
                known_face_labels.append(label)

    return known_face_encodings, known_face_labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    known_faces_folder = request.form['known_faces_folder']
    known_face_encodings, known_face_labels = load_known_faces(known_faces_folder)

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the captured frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        message = "Matched" if any(matches) else "Unmatched"

        # Draw rectangle and label on the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, message, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Release the webcam
    video_capture.release()

    # Encode image to base64 for displaying in HTML
    _, encoded_image = cv2.imencode('.jpg', frame)
    encoded_image = encoded_image.tobytes()

    return jsonify({'image': encoded_image.decode('utf-8'), 'message': message})

if __name__ == '__main__':
    app.run(debug=True)
