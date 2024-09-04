import cv2
import mediapipe as mp
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from gtts import gTTS
import io
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load the saved model
model = tf.keras.models.load_model('last_hope_model.h5')

# List of class labels corresponding to the classes the model was trained on
# class_labels = ['you', 'yes', 'thank you', 'ily', 'how', 'hello', 'are']
class_labels = ['are','hello','how','iloveyou','no','thankyou','yes','you']
def prepare_image(img):
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def speak_text(text):
    # Create gTTS object
    speech = gTTS(text=text, lang="en", slow=False, tld="com.au")
    
    # Save to in-memory buffer
    buffer = io.BytesIO()
    speech.write_to_fp(buffer)
    
    # Move buffer position to the beginning
    buffer.seek(0)
    
    # Load the buffer into pygame and play it directly
    pygame.mixer.music.load(buffer, "mp3")
    pygame.mixer.music.play()
    
    # Keep the script running while the audio plays
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def capture_hand_frames(capture_interval=2):
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 840)
    cap.set(4, 720)

    prev_frame_time = 0
    new_frame_time = 0
    last_capture_time = time.time()

    drawing_spec_connections = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green connections
    predicted_label = ''
    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        if not success:
            break

        h, w, c = img.shape
        white_background = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(white_background, landmarks, mp_hands.HAND_CONNECTIONS,
                                          drawing_spec_connections, drawing_spec_connections)

                for id, landmark in enumerate(landmarks.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    pts = [(cx, cy - 10), (cx + 10, cy), (cx, cy + 10), (cx - 10, cy)]
                    cv2.polylines(white_background, [np.array(pts)], isClosed=True, color=(42, 42, 165), thickness=4)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        
        # Capture frame and process if the interval has passed
        if time.time() - last_capture_time > capture_interval:
            # Use the white_background image for prediction
            img_array = prepare_image(white_background)
            predictions = model.predict(img_array)
            
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class]
            
            print(f'The predicted sign is: {predicted_label}')
            
            # Convert the predicted label to speech
            speak_text(predicted_label)
            
        
            last_capture_time = time.time()
        
        cv2.putText(white_background, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(white_background, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.putText(white_background, f"Predicted: {predicted_label}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow("Hand Detection", white_background)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_hand_frames(capture_interval=2)