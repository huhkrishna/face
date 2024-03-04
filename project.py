import cv2
import face_recognition
import numpy as np
import os
import streamlit as st

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

def match_faces(frame, known_face_encodings, known_face_labels):
    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Initialize the message as unmatched
    message = "Unmatched"
    matched = False

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if any(matches):
            label = known_face_labels[np.argmax(matches)]
            message = f"Matched"

            # Draw rectangle and label on the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, message, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            matched = True
            # Break out of the loop after processing the first face
            break

    return frame, message, matched

def main():
    st.title("Real-Time Face Recognition with Webcam")

    # Specify the relative path to your folder containing images of known faces
    known_faces_folder = st.text_input("Enter path to folder containing images of known faces:")

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    # Set video capture properties
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Load known faces and labels
    known_face_encodings, known_face_labels = load_known_faces(known_faces_folder)

    matched = False

    while True:
        # Read the next frame from the webcam
        ret, frame = video_capture.read()

        if not matched:
            # Match faces in the frame
            frame, message, matched = match_faces(frame, known_face_encodings, known_face_labels)

            # Display the resulting frame and the message
            st.image(frame, channels="BGR", use_column_width=True, caption=message)

        # Break the loop if 'q' key is pressed or a match is found
        if cv2.waitKey(1) & 0xFF == ord('q') or matched:
            break

    # Release the webcam and close all windows
    video_capture.release()

if __name__ == "__main__":
    main()