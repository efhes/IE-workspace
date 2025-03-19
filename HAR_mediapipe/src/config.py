import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from collections import namedtuple

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

class WindowMessage:
    def __init__(self, txt1=None, col1=None, pos1=None, txt2=None, col2=None, pos2=None, txt3=None, col3=None, pos3=None):

        # Initialize messages with default or provided values
        self.msg = []

        texts = [txt1, txt2, txt3]
        colors = [col1, col2, col3]
        positions = [pos1, pos2, pos3]

        for txt, col, pos in zip(texts, colors, positions):
            self.msg.append({
                'text': txt if txt is not None else '',
                'color': col if col is not None else (255, 0, 0),
                'position': pos if pos is not None else (0, 0)
            })
