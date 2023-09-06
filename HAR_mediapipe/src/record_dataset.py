import time
import cv2
import os
import libcamera
import numpy as np
from picamera2 import Picamera2, Preview

import sys
import tty
import termios

use_landmarks = True
if use_landmarks:
    import mediapipe as mp
    from landmarksLib import get_XYZ

# Array of class names
classes = ["Three", "Four", "Five"]

num_classes = len(classes)

# Base directory where the new dataset is going to be stored
dataset_dir = '../data/my_hands_dataset'

# Total number of images per class in the dataset
num_images_per_class = 10

# Percentage of the recorded images that will be used for training (rest for test)
training_percentage = 66 

green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)

def wait_for_keypress(target_key, exit_key, yes_to_all_key):
    print(f"Press '{target_key}' to continue or '{exit_key}' to exit... ('{yes_to_all_key}') if YES to all...")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    yes_to_all_result = False
    
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            char = sys.stdin.read(1)
            if char == target_key:
                break
            elif char == yes_to_all_key:
                print("Yes to all...")
                yes_to_all_result = True
                break
            elif char == exit_key:
                print("Exiting...")
                sys.exit(0)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return yes_to_all_result

def CreateDefaultDatasetFolders(base_dataset_dir):
    yes_to_all_result = False
    
    # Create default dataset folders
    for c in classes:
        new_dir = os.path.join(base_dataset_dir, "train", c)
        try:
            os.makedirs(new_dir)
        except:
            print('\n[WARNING]\n')
            print('The following directory ALREADY EXISTS!!!!')
            print(new_dir)
            print('\n[WARNING]\n')
            
            if not yes_to_all_result:
                yes_to_all_result = wait_for_keypress('c', 'q', 'y')
        
        new_dir = os.path.join(base_dataset_dir, "test", c)
        try:
            os.makedirs(new_dir)
        except:
            print('\n[WARNING]\n')
            print('The following directory ALREADY EXISTS!!!!')
            print(new_dir)
            print('\n[WARNING]\n')
            
            if not yes_to_all_result:
                yes_to_all_result = wait_for_keypress('c', 'q', 'y')

def DisplayPreviewScreen(picam2, class_to_record, hands=None):
    
    picam2.start()
    
    while True: #Empieza a mostrar la imagen por pantalla pra que le usuario se prepare. Cuando se presiona la tecla s el sistema compienza a grabar.
        image = picam2.capture_array("main")
        
        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if use_landmarks:
            # Process the image and get hand landmarks
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                image_rgb, landmark_values = get_XYZ(results, image_rgb)
                
        #image = cv2.flip(image,+1) #displays image in mirror format
        
        cv2.putText(image_rgb, 
                    "Class to be recorded: " + class_to_record,
                    org=(70,210), 
                    fontFace=2,
                    fontScale=0.75,
                    color=green)
        cv2.putText(image_rgb, "Get ready and press \'s\' to start recording",
                    org=(70,260), 
                    fontFace=2,
                    fontScale=0.75,
                    color=green)
        cv2.putText(image_rgb, "or \'q\' to exit.",
                    org=(70,300), 
                    fontFace=2,
                    fontScale=0.75,
                    color=green)
        
        cv2.imshow("Gesture Recorder", image_rgb)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("s"):
            picam2.stop()
            break
        elif key ==  ord('q'):
            exit()

def StartRecordingImages(picam2, num_images_to_record, hands=None):
    # num of images recorded counter
    i = 0
    recorded_images = []
    recording_frame = True
    
    picam2.start()
    
    started = time.time()
    last_save = started
    
    while True:
        # Read image 
        image = picam2.capture_array("main")
        
        # Convert the image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #image_rgb=cv2.flip(image_rgb,+1) #displays image in mirror format
                
        now = time.time()
        
        # We save only one image every second
        if now - last_save > 1: 
            image_rgb_nparray = np.ascontiguousarray(image_rgb, dtype=np.uint8)
            saved_image = np.copy(image_rgb_nparray)
            
            recorded_images.append(saved_image)
            last_save = now
            
            # We update the count of recorded images
            i+=1
        
        if now - last_save > 0.5: 
            # Get image dimensions
            height, width, _ = image_rgb.shape

            # Draw bounding rectangle
            bounding_rect = [(10, 10), (width-20, height-20)]
            cv2.rectangle(image_rgb, 
                        bounding_rect[0], 
                        bounding_rect[1], 
                        red, 
                        2)
       
        if use_landmarks:
            # Process the image and get hand landmarks
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                image_rgb, landmark_values = get_XYZ(results, image_rgb)
        
        #add annotations and change frame to color red
        cv2.putText(image_rgb, 
                    "Recording in progress...",
                    org=(20,40), 
                    fontFace=2,
                    fontScale=0.65,
                    color=red)
        cv2.putText(image_rgb, 
                    "Stored images: " + str(i+1) + "/" + str(num_images_to_record),
                    org=(20,80), 
                    fontFace=2,
                    fontScale=0.55,
                    color=red)
        
        cv2.imshow("Gesture Recorder", image_rgb)

        if i>=num_images_to_record:
            picam2.stop()
            time.sleep(1)
            break

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    return recorded_images 

def SaveRecordedImagesToDisk(recorded_images, class_to_save, num_samples_per_class):
    # Train images
    ini = 0
    end = num_samples_per_class['train']
    for i in range(ini, end):
        name = class_to_save + '_' + f'{i+1:05d}' + '.jpg'
        filename = os.path.join(dataset_dir, 'train', class_to_save, name)
        # save image in dataset
        cv2.imwrite(filename, recorded_images[i])
        print("Recorded ", filename)
        
    # Test images
    ini = num_samples_per_class['train']
    end = num_samples_per_class['train']+num_samples_per_class['test']
    for i in range(ini, end):
        name = class_to_save + '_' + f'{i+1:05d}' + '.jpg'
        filename = os.path.join(dataset_dir, 'test', class_to_save, name)
        # save image in dataset
        cv2.imwrite(filename, recorded_images[i])
        print("Recorded ", filename)
    
def main():
    num_samples_per_class = {}
    num_samples_per_class['train'] = int(num_images_per_class * training_percentage/100)
    num_samples_per_class['test']  = num_images_per_class - num_samples_per_class['train']
    num_samples_per_class['total'] = num_images_per_class

    print('\n[NEW DATASET RECORDING]')
    print('\t- %d classes: ' % num_classes, classes)
    print('\t- %d samples per class (%d samples in TOTAL)' % (num_samples_per_class['total'], num_classes * num_samples_per_class['total']))
    print('\t- %d samples for training (i.e. %0.2f%%) and %d for testing (i.e. %0.2f%%)' % (num_samples_per_class['train'], training_percentage, num_samples_per_class['test'], 100-training_percentage))
    
    CreateDefaultDatasetFolders(dataset_dir)
    #cv2.startWindowThread()

    highres_size = (1280, 720)
    large_size = (640, 480)
    small_size =  (320, 200)

    if use_landmarks:
        # Initialize Mediapipe
        print('Initializing Mediapipe')
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, 
                            max_num_hands=1, # max_num_hands=2, 
                            min_detection_confidence=0.5)
    else:
        hands = None
    
    # Initialize PiCamera
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": large_size}, 
                                                         controls={"AwbEnable": False,
                                                                   #"AwbMode": libcamera.controls.AwbModeEnum.Indoor,
                                                                   "AwbMode": libcamera.controls.AwbModeEnum.Auto,
                                                                   "AnalogueGain": 1.0})
    
    #preview_config = picam2.create_video_configuration(main = {"size": large_size, "format": "RGB888"})
    picam2.configure(preview_config)
    picam2.start_preview(Preview.NULL)
       
    # Main loop
    for c in classes:
        print("\nPREPARING TO RECORD CLASS:", c)
        
        DisplayPreviewScreen(picam2, c, hands=hands)
        
        recorded_images = StartRecordingImages(picam2, num_images_to_record=num_samples_per_class['total'], hands=hands)
        
        SaveRecordedImagesToDisk(recorded_images, c, num_samples_per_class)
            
    print('\n[RECORDING FINISHED SUCCESSFULLY!!!]\n')
    
    # Release resources
    cv2.destroyAllWindows()
    if use_landmarks:
        hands.close()
    picam2.close()

if __name__ == "__main__":
    main()

