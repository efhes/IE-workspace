import cv2
import mediapipe as mp
import libcamera
import numpy as np
from picamera2 import Picamera2, Preview
from keras.models import load_model, model_from_json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        landmark_values = []
        # Arrange the landmarkks as an array landmark_values[i] = [x, y]
        for landmark in hand_landmarks:
            landmark_values.append([landmark.x, 1-landmark.y])

    return annotated_image, landmark_values

def generate_mediapipe_result(image_rgb):
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    detection_result = detector.detect(mp_image)
    
    return detection_result

def get_XYZ(results, image_rgb):

    landmark_values = [[0, 0] for _ in range(21)]
    
    # esto para dibujar
    mp_drawing = mp.solutions.drawing_utils
    
    image_height, image_width, _ = image_rgb.shape

    for hand_landmarks in results.hand_landmarks:
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

def normalize_from_0_landmark(data):
  # We create a copy of the input data array to ensure that the original data is not modified,
  # and the modifications are made to the copy
  new_data = np.copy(data)
  # This loop iterates through each row of the data array.
  # Each row represents a different hand instance or sample.
  for i in range(data.shape[0]):
    # This condition helps to skip rows where the sum of X and Y coordinates is zero,
    # which may represent a case where there are no landmarks.
    if ((data[i, 0] + data[i, 1]) != 0):
      # These lines extract the X and Y coordinates of the center of the hand landmarks (typically, the palm).
      x_center = data[i, 0]
      y_center = data[i, 1]
      # This inner loop iterates through the remaining elements of the row,
      # starting from the third element (index 2).
      # In hand landmark data, these are typically the landmarks' X and Y coordinates, alternating.
      for k in range(2, data.shape[1], 2):
        # These lines calculate the new coordinates of each landmark relative to the center.
        # They subtract the center's X and Y coordinates from the corresponding landmark's X and Y coordinates.
        # This effectively shifts the coordinates so that the center becomes the new origin (0, 0).
        new_data[i, k] = data[i, k] - x_center
        new_data[i, k + 1] = data[i, k + 1] - y_center
  return new_data

debug_HAR = False
only_landmarks = False
#pred_mappings="ABCDEFGHIJKLMNOPQRSTUVWXYZ "
pred_mappings=['Five', 'Four', 'Three', 'Two', 'One']
landmark_values = [[0, 0] for _ in range(21)]

def main():
    large_size = (640, 480)
    small_size =  (320, 200)
    
    # Initialize PiCamera
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": large_size}, 
                                                         controls={"AwbEnable": False,
                                                                   #"AwbMode": libcamera.controls.AwbModeEnum.Indoor,
                                                                   "AwbMode": libcamera.controls.AwbModeEnum.Auto,
                                                                   "AnalogueGain": 1.0})
    picam2.configure(preview_config)
    picam2.start_preview(Preview.NULL)
    picam2.start()

    # Initialize Mediapipe
    print('Initializing Mediapipe')
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, 
                           max_num_hands=1, # max_num_hands=2, 
                           min_detection_confidence=0.5)
    
    # load model
    if only_landmarks == False:
        print("LOADING MODEL")

        #weight_path = '../models/poses.h5'
        #json_file = "../models/poses.json"
        weight_path = '../models/Five_Four_Three_FINE-TUNING2_L0.h5'
        json_file = "../models/Five_Four_Three_FINE-TUNING2_L0.json"

        with open(json_file) as f:
            loaded_model_json= f.read()

        model = model_from_json(loaded_model_json)
        model.load_weights(weight_path)

        print("LOADED MODEL:",json_file)

    print('Start processing frames')
    num_frames = 0
                
    while True:
        image = picam2.capture_array()
        num_frames=+1
        
        if debug_HAR:
            if num_frames % 25 == 0:
                print('num_frames = %d' % num_frames)
        
        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and get hand landmarks
        results = generate_mediapipe_result(image_rgb)
        
        print("\nresults", results, "\n")

        if results.hand_landmarks:
            if debug_HAR:
                print('HAY MANO')
            
            #image_rgb, landmark_values = get_XYZ(results, image_rgb)
            image_rgb, landmark_values = draw_landmarks_on_image(image_rgb, results)
        
        # Show the processed frame
        if only_landmarks:
            cv2.imshow("Hand Landmarks", image_rgb)
        else:
            if results.hand_landmarks:
                landmarks_nparray = np.array(landmark_values)
                landmarks_nparray = normalize_from_0_landmark(landmarks_nparray)
                landmarks_nparray = np.reshape(landmarks_nparray, (1,21,2,1))
                pred = pred_mappings[np.argmax(model(landmarks_nparray))]
                cv2.putText(image_rgb, 
                            "Predicted class: "+pred,
                            org=(10, large_size[1]-20), 
                            fontFace=2,
                            fontScale=0.75,
                            color=(0,200,100))
            else:
                cv2.putText(image_rgb, 
                            "Predicted class: none",
                            org=(10, large_size[1]-20), 
                            fontFace=2,
                            fontScale=0.75,
                            color=(0,200,100))
            cv2.imshow("Model prediction", image_rgb)
        
        # Press 'q' to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()
    hands.close()
    picam2.close()

if __name__ == "__main__":
    main()
