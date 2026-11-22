print("Loading dependencies... (It might take a while the first time)")
from pathlib import Path
import keras
import cv2
import os
import sys
import time
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent

root_path = os.getcwd()
path_har = os.path.join(root_path, "FER_mediapipe", "src")
path_common = os.path.join(root_path, "common")

# 3. Añadirlas al path y verificar si existen
for p in [path_har, path_common]:
    if p not in sys.path:
        sys.path.append(p)
    if not os.path.exists(p):
        print(f"⚠️ ¡OJO! La ruta no existe: {p}")
    else:
        print(f"✅ Ruta añadida: {p}")

# 4. Intentar la importación
try:
    import landmarks_utils
    print("🚀 landmarks_utils importado con éxito")
except ModuleNotFoundError as e:
    print(f"❌ Error: {e}")

# Auto-detect Raspberry Pi so the script runs seamlessly on both Mac/PC and Raspberry Pi
def _is_raspberry_pi():
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r", errors="ignore") as f:
                return "raspberry pi" in f.read().lower()
    except Exception:
        pass
    return False

ON_RASPBERRY_PI = _is_raspberry_pi()
ON_SENSE_HAT = ON_RASPBERRY_PI

from cameras import CVCamera, PICamera, CameraConfig
from config import Config
from config import ConfigMediapipeDetector, RecordingSetup
from gui import Colors, WindowMessage

from landmarks_utils import (
    GetFaceLandmarksListFromDetectionResult,
    draw_landmarks_on_image,
    FlattenFaceLandmarks,
    ArrangeInputDataForNetwork,
    normalize_L0,
    normalize_size,
    normalize_rotation,
)

if ON_RASPBERRY_PI:
    if ON_SENSE_HAT:
        from sense_hat import SenseHat

if ON_RASPBERRY_PI:
    cam_config = CameraConfig(FPS=30, resolution='large')
else:
    cam_config = CameraConfig(FPS=30, resolution='highres')

# Instantiate the configuration
classes=['angry', 'happy', 'sad', 'surprise']
window_title = "Face expressions recognition demonstrator"
colors = Colors()
colors.SelectRandomColorFromListForClasses(classes)
config = Config(classes=classes, use_landmarks = True)

# FRAME RATE
FPS = 15

MODEL_PATH = PROJECT_DIR / "models" / "new_model_CNN_L0_size_rotation.keras"
if not MODEL_PATH.is_file():
    MODEL_PATH = PROJECT_DIR / "models" / "new_model_CNN_L0_size.keras"
#MODEL_PATH = PROJECT_DIR / "models" / "FER_finetuned.keras"

# Reference aspect ratio of the training dataset (1280x720 = 16:9)
TRAINING_ASPECT_RATIO = 1280.0 / 720.0

# Remember they must keep the same order than the used in training.
# The image is used when using the sense hat. It must be a mask image of 8x8 px.
classes = [
    {"label": "angry", "image": "hmmm.png"},
    {"label": "happy", "image": "smiley.png"},
    {"label": "sad", "image": "sad.png"},
    {"label": "surprise", "image": "surprise.png"},
]

def get_face_corners(landmarks, image_size):
    maxs = np.squeeze(np.max(landmarks, axis=0))
    mins = np.squeeze(np.min(landmarks, axis=0))
    corner_ul = np.array([mins[0], 1 - maxs[1]])
    corner_br = np.array([maxs[0], 1 - mins[1]])

    for i in range(2):
        corner_ul[i] = int(corner_ul[i] * image_size[i])
        corner_br[i] = int(corner_br[i] * image_size[i])

    corner_ul = corner_ul.astype(np.int32)
    corner_br = corner_br.astype(np.int32)
    return corner_ul, corner_br


def draw_results(image, results, classes, face_corners=None, sense_hat=None):
    """
    Draws the results and graphics on top of the camera's recording.

    Parameters:
        - image: array of the recorded image
        - results: normalized logits obtained from the model
        - face_corners: list with the corners with format [corner_up_left, corner_down_right] == [(x0,y0), (x1,y1)]
    """
    if face_corners[0] == face_corners[1]:
        return image

    prediction_idx = np.argmax(results)
    probability = results[0, prediction_idx] * 100
    label = classes[prediction_idx]["label"]

    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    if ON_RASPBERRY_PI:
        text_font = ImageFont.truetype("NotoMono-Regular.ttf", 16)
    else:
        # This is the font that is showed in the demo.
        # Asjust it to one font of your system.
        text_font = ImageFont.truetype("Arial.ttf", 16)

    draw.text(
        (face_corners[0][0], face_corners[0][1] - 20),
        f"Emotion: {label} ({probability:.2f}%)",
        font=text_font,
        fill=(0, 0, 255, 255),
    )

    if face_corners is not None:
        draw.rectangle(face_corners, outline=(0, 0, 255, 255))

    if ON_SENSE_HAT:
        if sense_hat is not None:
            sense_hat.load_image(
                os.path.join("emoticons", classes[prediction_idx]["image"])
            )

    image_rgb = np.array(image_pil)
    return image_rgb


def main():
    # Load model to use
    print("Loading model...")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded!")

    # Create the detector
    detector = ConfigMediapipeDetector('FER_mediapipe/models/face_landmarker.task')

    # Start camera, use CVCamera if working on a laptop and PICamera in case you are working on a Raspberry PI
    if ON_RASPBERRY_PI:
        cam = PICamera(recording_res=cam_config.resolution)
        if ON_SENSE_HAT:
            sense_hat = SenseHat()
            sense_hat.set_rotation(0)
    else:
        cam = CVCamera(recording_res=cam_config.resolution, index_cam=0)
        #cam = CVCamera(recording_res=cam_config.resolution, index_cam=0) # index_cam=1 is for the external camera, index_cam=0 is for the internal camera
        sense_hat = None

    cam.start()

    now = 0
    last = 0
    predictions = np.zeros((1, len(classes)))
    prediction_idx = 0
    pred = 'None'
    probability = 0.0
    
    while True:
        # Load the input image (both PICamera "RGB888" and CVCamera return BGR frames for OpenCV)
        image = cam.read_frame()
        
        if image is None:
            # Depending the setup, the camera might need approval to activate, so wait until we start receiving images.
            print("Waiting for camera input")
            continue

        # Get current time to control the processing rate of the model
        now = time.time()
        
        # Always convert the BGR camera frame to RGB before feeding it to MediaPipe (SRGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and get face landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)
        
        landmark_values = []
        
        # Test if we have successfully detected a face in the image. Keep compatibility
        # with both current and legacy MediaPipe APIs.
        face_landmarks_list = getattr(detection_result, "face_landmarks", None)
        if face_landmarks_list is None and hasattr(detection_result, "multi_face_landmarks"):
            face_landmarks_list = detection_result.multi_face_landmarks

        if detection_result is None or face_landmarks_list is None or len(face_landmarks_list) == 0:
            pred = 'None'
            probability = 0.0
            predictions = np.zeros((1, len(classes)))
        else:
            landmark_values = GetFaceLandmarksListFromDetectionResult(detection_result)
        
            if not landmark_values:
                pred = 'None'
                probability = 0.0
                predictions = np.zeros((1, len(classes)))
            else:
                # Draw detected face mesh on the display frame
                image = draw_landmarks_on_image(image, detection_result)

                # Process the landmarks through the model every 0.25s
                if now - last > 0.25:
                    flat_landmark_values = np.asarray(FlattenFaceLandmarks(landmark_values), dtype=np.float32)

                    # Correct aspect-ratio difference when running at 4:3 (640x480 on Raspberry Pi)
                    # vs 16:9 (1280x720 in training dataset / Mac) so normalized x/y proportions match
                    frame_h, frame_w = image.shape[:2]
                    current_aspect = float(frame_w) / float(frame_h)
                    if abs(current_aspect - TRAINING_ASPECT_RATIO) > 1e-3:
                        flat_landmark_values[0::2] *= (current_aspect / TRAINING_ASPECT_RATIO)

                    # Apply geometric normalizations (translation + scale/camera-distance + in-plane rotation)
                    landmarks = normalize_L0(flat_landmark_values)
                    landmarks = normalize_size(landmarks)
                    landmarks = normalize_rotation(landmarks)

                    # Reshape to (1, 478, 2, 1) for the network
                    landmarks = np.expand_dims(landmarks, axis=0)  # (1, 956)
                    landmarks = ArrangeInputDataForNetwork(landmarks)  # (1, 478, 2, 1)
                    
                    # Run inference
                    predictions = np.asarray(model(landmarks, training=False))
                    
                    if predictions is not None:
                        prediction_idx = int(np.argmax(predictions))
                        probability = float(predictions[0, prediction_idx]) * 100.0
                        pred = config.classes[prediction_idx]
                        print(f"Predicted emotion: {pred} ({probability:.2f}%)")
                    else:
                        print("No predictions available.")

                    last = time.time()

        if ON_RASPBERRY_PI and ON_SENSE_HAT and sense_hat is not None:
            if pred == 'None':
                sense_hat.clear()
            else:
                sense_hat.load_image(
                    os.path.join("./FER_mediapipe/emoticons", classes[prediction_idx]["image"])
                )
                
        if pred == 'None':
            color1 = colors.color['black']
        else:
            color1 = colors.GetColorForClass(pred)
            
        class_msgs = WindowMessage(
            txt1 = "Predicted class: " + pred + " (%0.2f)" % probability, pos1 = (10, cam_config.resolution[1]-20), col1 = color1,
            txt2 = "", pos2 = (0, 0), col2 = colors.color['black'],
            txt3 = "", pos3 = (0, 0), col3 = colors.color['black'])

        class_msgs.ShowWindowMessages(image)
        
        cv2.imshow(window_title, image)
        key = cv2.waitKey(int(1 / FPS * 1000)) & 0xFF
        if key == ord("q"):
            if ON_RASPBERRY_PI:
                if ON_SENSE_HAT and sense_hat is not None:
                    sense_hat.clear()
            break

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
