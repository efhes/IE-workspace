import mediapipe as mp

def get_XYZ(results, image_rgb):

    landmark_values = [[0, 0] for _ in range(21)]
    
    # esto para dibujar
    mp_drawing = mp.solutions.drawing_utils
    
    image_height, image_width, _ = image_rgb.shape
        
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(image_rgb, 
                                                  hand_landmarks, 
                                                  mp.solutions.hands.HAND_CONNECTIONS,
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=5),
                                                  mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=4))
        
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x  # * image_width
            y = 1 - hand_landmarks.landmark[i].y  # * image_height (1-y for providing same shape as in image)
            landmark_values[i] = [x, y]
                
    return image_rgb, landmark_values

