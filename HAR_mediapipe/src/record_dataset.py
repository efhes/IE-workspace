import cv2
import os
import time
import numpy as np
import mediapipe as mp

from cameras import CVCamera, PICamera
from config import Config, CameraConfig, Colors
from config import ConfigMediapipeDetector, RecordingSetup
from gui import WindowMessage
from landmarksLib import draw_landmarks_on_image

ON_RASPBERRY_PI = False

# Instantiate the configuration
window_title = "Hand gestures recorder"
colors = Colors()
config = Config(classes=['three', 'four', 'five'], num_images_per_class=5, use_landmarks = True)
cam_config = CameraConfig(FPS=15, resolution='highres')

def DisplayPreviewScreen(cam, detector, messages=None):
    #cam.start()
    
    while True: #Empieza a mostrar la imagen por pantalla pra que le usuario se prepare. Cuando se presiona la tecla s el sistema compienza a grabar.
        #image = cam.capture_array("main")
        image = cam.read_frame()
        if image is None:
            # Depending the setup, the camera might need approval to activate, so wait until we start receiving images.
            print("Waiting for camera input")
            continue

        if config.use_landmarks:
            # Convert the image to RGB for Mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    

            # Process the image and get hand landmarks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)

            if detection_result.hand_landmarks:
                #image_rgb, landmark_values = get_XYZ(results, image_rgb)
                image, landmark_values = draw_landmarks_on_image(image, detection_result)
                
        #image = cv2.flip(image,+1) #displays image in mirror format
        
        messages.ShowWindowMessages(image)
        
        cv2.imshow(window_title, image)
        
        key = cv2.waitKey(int(1 / cam_config.FPS * 1000)) & 0xFF
        
        if key == ord("s"):
            #cam.stop()
            break
        elif key ==  ord('q'):
            exit()

def StartRecordingImages(cam, detector, num_images_to_record, messages=None):
    # num of images recorded counter
    n_recorded = 0
    recorded_images = []
    recording_frame = True
    
    #cam.start()
    
    started = time.time()
    last_save = started
    
    while True:
        # Read image from camera
        image = cam.read_frame()
        if image is None:
            # Depending the setup, the camera might need approval to activate, so wait until we start receiving images.
            print("Waiting for camera input")
            continue
        
        #image_rgb=cv2.flip(image_rgb,+1) #displays image in mirror format
                
        now = time.time()
        
        # We save only one image every second
        if now - last_save > 1: 
            image_rgb_nparray = np.ascontiguousarray(image, dtype=np.uint8)
            saved_image = np.copy(image_rgb_nparray)
            
            recorded_images.append(saved_image)
            last_save = now
            
            # We update the count of recorded images
            n_recorded+=1
        
        if now - last_save > 0.5: 
            # Get image dimensions
            height, width, _ = image_rgb.shape

            # Draw bounding rectangle
            bounding_rect = [(10, 10), (width-20, height-20)]
            cv2.rectangle(image_rgb, bounding_rect[0], bounding_rect[1], colors.color['red'], 2)
       
        if config.use_landmarks:
            # Convert the image to RGB for Mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and get hand landmarks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)

            if detection_result.hand_landmarks:
                #image_rgb, landmark_values = get_XYZ(results, image_rgb)
                image, landmark_values = draw_landmarks_on_image(image, detection_result)
        
        messages.msg[1]['text'] = "Stored images: " + str(n_recorded + 1) + "/" + str(num_images_to_record)
        messages.ShowWindowMessages(image)

        cv2.imshow(window_title, image)

        if n_recorded>=num_images_to_record:
            #cam.stop()
            time.sleep(1)
            break

        #key = cv2.waitKey(1) & 0xFF
        key = cv2.waitKey(int(1 / cam_config.FPS * 1000)) & 0xFF

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
        filename = os.path.join(config.dataset_dir, 'train', class_to_save, name)
        # save image in dataset
        cv2.imwrite(filename, recorded_images[i])
        print("Recorded ", filename)
        
    # Test images
    ini = num_samples_per_class['train']
    end = num_samples_per_class['train']+num_samples_per_class['test']
    for i in range(ini, end):
        name = class_to_save + '_' + f'{i+1:05d}' + '.jpg'
        filename = os.path.join(config.dataset_dir, 'test', class_to_save, name)
        # save image in dataset
        cv2.imwrite(filename, recorded_images[i])
        print("Recorded ", filename)

def main():
    # Recording setup 
    num_samples_per_class = RecordingSetup(config)

    # Create default dataset folders
    config.CreateDefaultDatasetFolders()

    # Create the detector
    detector = ConfigMediapipeDetector()

    # Start camera, use CVCamera if working on a laptop and PICamera in case you are working on a Raspberry PI
    if ON_RASPBERRY_PI:
        cam = PICamera(recording_res=cam_config.resolution)
        sense_hat = SenseHat()
        sense_hat.set_rotation(180)
    else:
        cam = CVCamera(recording_res=cam_config.resolution, index_cam=1)
        sense_hat = None

    # Start camera
    cam.start()

    # Main loop
    for c in config.classes:
        print("\nPREPARING TO RECORD CLASS:", c)

        preview_msgs = WindowMessage(
            txt1 = "Class to be recorded: " + c, pos1 = (70, 210), col1 = colors.color['green'],
            txt2 = "Get ready and press \'s\' to start recording", pos2 = (70, 260), col2 = colors.color['green'],
            txt3 = "or \'q\' to exit.", pos3 = (70, 300), col3 = colors.color['green'])
        
        # Display preview screen
        DisplayPreviewScreen(cam, detector, preview_msgs)

        recording_msgs = WindowMessage(
            txt1 = "Recording in progress...", pos1 = (20, 40), col1 = colors.color['red'],
            txt2 = "Get ready and press \'s\' to start recording", pos2 = (20, 80), col2 = colors.color['red'],
            txt3 = "", pos3 = (20, 120), col3 = colors.color['red'])
        
        # Start recording images
        recorded_images = StartRecordingImages(cam, detector, num_samples_per_class['total'], recording_msgs)

        # Save images to disk
        SaveRecordedImagesToDisk(recorded_images, c, num_samples_per_class)

    print('\n[RECORDING FINISHED SUCCESSFULLY!!!]\n')

    post_msgs = WindowMessage(
            txt1 = "Recording finished successfully!!!", pos1 = (70, 210), col1 = colors.color['blue'],
            txt2 = "You can now press \'q\' to exit...", pos2 = (70, 260), col2 = colors.color['blue'],
            txt3 = "", pos3 = (70, 300), col3 = colors.color['blue'])

    # Display final message
    DisplayPreviewScreen(cam, detector, post_msgs)

    # Release resources
    cam.stop()
if __name__ == "__main__":
    main()

       

    
#    # Release resources
#    cv2.destroyAllWindows()
#    if use_landmarks:
#        hands.close()
#    picam2.close()

