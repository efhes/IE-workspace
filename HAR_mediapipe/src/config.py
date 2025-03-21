import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from gui import wait_for_keypress
import random

# Configuration class for the dataset generation
class Config:
    def __init__(self, 
                 classes=None, 
                 dataset_dir='./data/new_dataset/', 
                 num_images_per_class=100, 
                 training_percentage=66, 
                 use_landmarks=True,
                 save_images=False):
        
        # Use landmarks to record images
        self.use_landmarks = use_landmarks
        
        # Number of landmarks
        self.num_landmarks = 21

        # Array of class names
        self.classes = classes if classes is not None else ["gesture1", "gesture2", "gesture3"]
        self.num_classes = len(self.classes)

        # Base directory where the new dataset is going to be stored
        self.dataset_dir = dataset_dir

        # Total number of images per class in the dataset
        self.num_images_per_class = num_images_per_class

        # Percentage of the recorded images that will be used for training (rest for test)
        self.training_percentage = training_percentage

        # Flag to save images with landmarks
        self.save_images = save_images
    
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
        self.color['black'] = (0, 0, 0)
        self.color['yellow'] = (0, 255, 255)
        self.color['white'] = (255, 255, 255)
        self.color['purple'] = (128, 0, 128)
        self.color['orange'] = (0, 165, 255)
        self.color['gray'] = (128, 128, 128)
        self.color['brown'] = (42, 42, 165)
        self.color['pink'] = (147, 20, 255)
        self.color['cyan'] = (255, 255, 0)
        self.color['magenta'] = (255, 0, 255)
        self.color['light_blue'] = (255, 216, 150)
        self.color['light_green'] = (250, 250, 211)
        self.color['light_red'] = (0, 0, 204)
        self.color['light_yellow'] = (0, 204, 204)
        self.color['light_purple'] = (153, 0, 153)
        self.color['light_orange'] = (0, 128, 255)
        self.color['light_gray'] = (192, 192, 192)
        self.color['light_brown'] = (42, 107, 165)
        self.color['light_pink'] = (204, 153, 255)
        self.color['light_cyan'] = (255, 255, 204)
        self.color['light_magenta'] = (255, 204, 255)

        self.classes = []
        self.class_colors = []

    def SelectRandomColorFromListForClasses(self, classes):
        self.classes = classes
        self.class_colors = []
        for _ in range(len(classes)):
            self.class_colors.append(self.color[random.choice(list(self.color.keys()))])
    
    def GetColorForClass(self, class_name):
        return self.class_colors[self.classes.index(class_name)]


def ConfigMediapipeDetector(model_path='./models/hand_landmarker.task'):
    base_options = python.BaseOptions(model_asset_path=model_path)
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
