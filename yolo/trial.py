import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

prev_frame_time = 0
new_frame_time = 0

# Custom drawing styles for connections (lines)
drawing_spec_connections = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green connections

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

    # Create a white background
    h, w, c = img.shape
    white_background = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(img_rgb)

    # Draw hand landmarks with custom colors on the white background
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections with green lines
            mp_drawing.draw_landmarks(white_background, landmarks, mp_hands.HAND_CONNECTIONS, 
                                      drawing_spec_connections, drawing_spec_connections)
            
            # Draw red diamonds instead of circles for each landmark
            for id, landmark in enumerate(landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # Define the points for the diamond
                pts = [
                    (cx, cy - 5),  # Top point
                    (cx + 5, cy),  # Right point
                    (cx, cy + 5),  # Bottom point
                    (cx - 5, cy)   # Left point
                ]
                
                # Draw the diamond
                cv2.polylines(white_background, [np.array(pts)], isClosed=True, color=(42, 42, 165), thickness=2)  # Red diamond

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Display FPS on the image
    cv2.putText(white_background, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Create a dialogue box
    dialogue_box_height = 100
    dialogue_box_color = (0, 0, 0)  # Black
    dialogue_box_thickness = -1  # Fill the rectangle
    cv2.rectangle(white_background, (0, h - dialogue_box_height), (w, h), dialogue_box_color, dialogue_box_thickness)

    # Add text to the dialogue box
    text = "Hand Detection Active"
    text_color = (0, 255, 0)  # Green
    cv2.putText(white_background, text, (10, h - int(dialogue_box_height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

    # Display the white background with drawn landmarks
    cv2.imshow("Hand Detection", white_background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()