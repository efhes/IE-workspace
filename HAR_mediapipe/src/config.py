import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gui import wait_for_keypress

# Configuration class for the dataset generation
class Config:
    def __init__(self, 
                 classes=None, 
                 dataset_dir='./data/new_dataset/', 
                 num_images_per_class=100, 
                 training_percentage=66, 
                 use_landmarks=True):
        
        # Use landmarks to record images
        self.use_landmarks = use_landmarks
        
        # Array of class names
        self.classes = classes if classes is not None else ["gesture1", "gesture2", "gesture3"]
        self.num_classes = len(self.classes)

        # Base directory where the new dataset is going to be stored
        self.dataset_dir = dataset_dir

        # Total number of images per class in the dataset
        self.num_images_per_class = num_images_per_class

        # Percentage of the recorded images that will be used for training (rest for test)
        self.training_percentage = training_percentage
    
    def CreateDefaultDatasetFolders(self):
        yes_to_all_result = False
        
        # Create default dataset folders
        for c in self.classes:
            new_dir = os.path.join(self.dataset_dir, "train", c)
            try:
                os.makedirs(new_dir)
            except:
                print('\n[WARNING]\n')
                print('The following directory ALREADY EXISTS!!!!')
                print(new_dir)
                print('\n[WARNING]\n')
                
                if not yes_to_all_result:
                    yes_to_all_result = wait_for_keypress('c', 'q', 'y')
            
            new_dir = os.path.join(self.dataset_dir, "test", c)
            try:
                os.makedirs(new_dir)
            except:
                print('\n[WARNING]\n')
                print('The following directory ALREADY EXISTS!!!!')
                print(new_dir)
                print('\n[WARNING]\n')
                
                if not yes_to_all_result:
                    yes_to_all_result = wait_for_keypress('c', 'q', 'y')

class CameraConfig:
    def __init__(self, FPS=30, resolution='highres'): 
        # RESOLUTIONS
        self.resolutions = {}
        self.resolutions['highres'] = (1280, 720)
        self.resolutions['large'] = (640, 480)
        self.resolutions['small'] = (320, 200)

        # RESOLUTION
        self.resolution = self.resolutions[resolution]

        # FRAME RATE
        self.FPS = FPS

class Colors:
    def __init__(self):
        #background=(0, 0, 0), text=(255, 255, 255), button=(0, 0, 255)):
        #self.background = background
        #self.text = text
        #self.button = button
        # Colors for annotations
        self.color = {}
        self.color['green'] = (0, 255, 0)
        self.color['blue'] = (255, 0, 0)
        self.color['red'] = (0, 0, 255)

def ConfigMediapipeDetector():
    base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, min_tracking_confidence=0.5)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector


def RecordingSetup(config):
    num_samples_per_class = {}
    num_samples_per_class['train'] = int(config.num_images_per_class * config.training_percentage/100)
    num_samples_per_class['test']  = config.num_images_per_class - num_samples_per_class['train']
    num_samples_per_class['total'] = config.num_images_per_class

    print('\n[NEW DATASET RECORDING]')
    print('\t- %d classes: ' % config.num_classes, config.classes)
    print('\t- %d samples per class (%d samples in TOTAL)' % 
          (num_samples_per_class['total'], 
           config.num_classes * num_samples_per_class['total']))
    print('\t- %d samples for training (i.e. %0.2f%%) and %d for testing (i.e. %0.2f%%)' % 
          (num_samples_per_class['train'], 
           config.training_percentage, 
           num_samples_per_class['test'], 
           100-config.training_percentage))
    return num_samples_per_class
