import cv2
import time
import numpy as np
import mediapipe as mp
import keras

from cameras import CVCamera, PICamera
from config import Config, CameraConfig, Colors
from config import ConfigMediapipeDetector
from gui import WindowMessage
from landmarksLib import draw_landmarks_on_image

ON_RASPBERRY_PI = False
MODEL_PATH = "models/Five_Four_Three_CNN1.keras"

# Instantiate the configuration
# Classes to be recognized; ATENTION: 'None' class must be the last one; the others must be specified in the order they were trained (alphabetical order)
classes=['Five', 'Four', 'Three', 'None']
window_title = "Hand gestures recognition demonstrator"
colors = Colors()
colors.SelectRandomColorFromListForClasses(classes)
config = Config(classes=classes, use_landmarks = True)
cam_config = CameraConfig(FPS=15, resolution='highres')

debug_HAR = False
only_landmarks = False
pred_mappings=classes
landmark_values = [[0, 0] for _ in range(21)]

def main():
    # Create the detector
    detector = ConfigMediapipeDetector('./models/hand_landmarker.task')

    # Start camera, use CVCamera if working on a laptop and PICamera in case you are working on a Raspberry PI
    if ON_RASPBERRY_PI:
        cam = PICamera(recording_res=cam_config.resolution)
        sense_hat = SenseHat()
        sense_hat.set_rotation(180)
    else:
        cam = CVCamera(recording_res=cam_config.resolution, index_cam=1)
        sense_hat = None
    
    
    # Load model to use
    print("Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded!")

    # Start camera
    cam.start()
    
    now = 0
    last = 0
    num_frames = 0 # Number of frames processed
    pred = 'None'

    while True:
        image = cam.read_frame()
        if image is None:
            # Depending the setup, the camera might need approval to activate, so wait until we start receiving images.
            print("Waiting for camera input")
            continue

        now = time.time()
        num_frames=+1
        
        if debug_HAR:
            if num_frames % 25 == 0:
                print('num_frames = %d' % num_frames)
        
        
        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    

        # Process the image and get hand landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            image, landmark_values = draw_landmarks_on_image(image, detection_result)
            
            if debug_HAR:
                print('HAY MANO')
                print("\nresults", landmark_values, "\n")
            
            landmarks_nparray = np.array(landmark_values)
            landmarks_nparray = np.reshape(landmarks_nparray, (1,21,2,1))
            
            # Process the image to the model every 0.25s
            if now - last > 0.25: #and landmarks.sum() != 0:
                predictions = model(landmarks_nparray)
                pred = pred_mappings[np.argmax(predictions)]
                conf = np.max(predictions)
                
                if debug_HAR:
                    # We print the confidence of the winning prediction
                    print('Prediction:', pred, 'Confidence:', conf)

                last = time.time()
        else:
            pred = 'None'
            predictions = np.zeros((1, len(classes)))
            conf = 1.0
        
        class_msgs = WindowMessage(
            txt1 = "Predicted class: " + pred + " (%0.2f)" % conf, pos1 = (10, cam_config.resolution[1]-20), col1 = colors.GetColorForClass(pred),
            txt2 = "", pos2 = (0, 0), col2 = colors.color['black'],
            txt3 = "", pos3 = (0, 0), col3 = colors.color['black'])

        class_msgs.ShowWindowMessages(image)

        cv2.imshow(window_title, image)

        # Press 'q' to quit the program
        key = cv2.waitKey(int(1 / cam_config.FPS * 1000)) & 0xFF
        
        if key ==  ord('q'):
            if ON_RASPBERRY_PI:
                sense_hat.clear()
            break
    
    # Release resources
    cam.stop()
    exit()

if __name__ == "__main__":
    main()
