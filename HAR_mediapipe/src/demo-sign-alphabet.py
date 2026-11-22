from pathlib import Path
import cv2
import time
import numpy as np
import mediapipe as mp
import keras
from keras.models import load_model, model_from_json
import sys
import os

PROJECT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_DIR.parent

path_har = str(PROJECT_DIR / "src")
path_common = str(WORKSPACE_DIR / "common")

for p in [path_har, path_common]:
    if p not in sys.path:
        sys.path.append(p)
    if not os.path.exists(p):
        print(f"⚠️ ¡OJO! La ruta no existe: {p}")
    else:
        print(f"✅ Ruta añadida: {p}")


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

if ON_RASPBERRY_PI and ON_SENSE_HAT:
    from sense_hat import SenseHat

from cameras import CVCamera, PICamera, CameraConfig
from config import Config, ConfigMediapipeDetector
from gui import Colors, WindowMessage
from landmarksLib import draw_landmarks_on_image

MODEL_JSON_PATH = PROJECT_DIR / "models" / "poses.json"
MODEL_WEIGHTS_PATH = PROJECT_DIR / "models" / "poses.h5"
MODEL_KERAS_PATH = PROJECT_DIR / "models" / "poses.keras"
DETECTOR_PATH = str(PROJECT_DIR / "models" / "hand_landmarker.task")

# 26 sign-language alphabet letters + space (' ')
pred_mappings = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
window_title = "Sign Alphabet Recognition Demonstrator"
colors = Colors()
colors.SelectRandomColorFromListForClasses(pred_mappings)
config = Config(classes=pred_mappings, use_landmarks=True)

if ON_RASPBERRY_PI:
    cam_config = CameraConfig(FPS=15, resolution='large')
else:
    cam_config = CameraConfig(FPS=15, resolution='highres')

debug_HAR = False
only_landmarks = False


def load_poses_model():
    print("LOADING MODEL...")
    if MODEL_JSON_PATH.is_file() and MODEL_WEIGHTS_PATH.is_file():
        with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
            loaded_model_json = f.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(str(MODEL_WEIGHTS_PATH))
        print(f"LOADED MODEL: {MODEL_JSON_PATH} + {MODEL_WEIGHTS_PATH}")
        return model
    elif MODEL_KERAS_PATH.is_file():
        model = load_model(MODEL_KERAS_PATH)
        print(f"LOADED MODEL: {MODEL_KERAS_PATH}")
        return model
    else:
        raise FileNotFoundError(f"Could not find poses model in {PROJECT_DIR / 'models'}")


def main():
    detector = ConfigMediapipeDetector(DETECTOR_PATH)

    if ON_RASPBERRY_PI:
        cam = PICamera(recording_res=cam_config.resolution)
        if ON_SENSE_HAT:
            sense_hat = SenseHat()
            sense_hat.set_rotation(0)
        else:
            sense_hat = None
    else:
        cam = CVCamera(recording_res=cam_config.resolution, index_cam=0)
        sense_hat = None

    model = None
    if not only_landmarks:
        model = load_poses_model()

    cam.start()

    now = 0.0
    last = 0.0
    num_frames = 0
    pred = 'None'
    conf = 0.0

    print("Start processing frames")

    while True:
        image = cam.read_frame()
        if image is None:
            print("Waiting for camera input")
            continue

        now = time.time()
        num_frames += 1

        if debug_HAR and num_frames % 25 == 0:
            print('num_frames = %d' % num_frames)

        # Convert BGR frame to RGB for MediaPipe (mediapipe 1.2.3 Tasks API)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            image, _ = draw_landmarks_on_image(image, detection_result)

            # Extract the first hand's 21 landmarks using (x, 1 - y) convention matching poses.h5 training
            first_hand = detection_result.hand_landmarks[0]
            landmark_values = [[float(lm.x), float(1.0 - lm.y)] for lm in first_hand[:21]]

            if debug_HAR:
                print('HAY MANO')
                print("\nresults", landmark_values, "\n")

            if not only_landmarks and (now - last > 0.25):
                landmarks_nparray = np.asarray(landmark_values, dtype=np.float32).reshape((1, 21, 2, 1))
                predictions = np.asarray(model(landmarks_nparray, training=False))
                pred_idx = int(np.argmax(predictions))
                pred = pred_mappings[pred_idx]
                conf = float(np.max(predictions)) * 100.0

                if debug_HAR:
                    print(f"Prediction: {pred} ({conf:.2f}%)")

                last = time.time()
        else:
            pred = 'None'
            conf = 0.0

        if only_landmarks:
            cv2.imshow(window_title, image)
        else:
            color1 = colors.color['black'] if pred == 'None' else colors.GetColorForClass(pred)
            class_msgs = WindowMessage(
                txt1=f"Predicted class: {pred} ({conf:.2f}%)",
                pos1=(10, cam_config.resolution[1] - 20),
                col1=color1,
                txt2="", pos2=(0, 0), col2=colors.color['black'],
                txt3="", pos3=(0, 0), col3=colors.color['black'],
            )
            class_msgs.ShowWindowMessages(image)
            cv2.imshow(window_title, image)

        key = cv2.waitKey(int(1 / cam_config.FPS * 1000)) & 0xFF
        if key == ord('q'):
            if ON_RASPBERRY_PI and ON_SENSE_HAT and sense_hat is not None:
                sense_hat.clear()
            break

    cam.stop()
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
