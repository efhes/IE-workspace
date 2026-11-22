import os
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

def hand_get_XYZ(results, image_rgb):

    landmark_values = [[0, 0] for _ in range(21)]

    # esto para dibujar
    mp_drawing = mp.solutions.drawing_utils

    image_height, image_width, _ = image_rgb.shape

    for hand_landmarks in results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            image_rgb,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(
                color=(255, 255, 0), thickness=4, circle_radius=5
            ),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=4),
        )

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x  # * image_width
            y = (
                1 - hand_landmarks.landmark[i].y
            )  # * image_height (1-y for providing same shape as in image)
            landmark_values[i] = [x, y]

    return image_rgb, landmark_values


def face_get_XYZ(results, image_rgb=None):
    """
    Extract face landmarks from MediaPipe and return them as a (478, 2) array.

    Supports both the legacy API (`results.multi_face_landmarks`) and the current
    FaceLandmarker API (`results.face_landmarks` from FaceLandmarkerResult).
    """

    landmarks = np.zeros((478, 2), dtype=np.float32)

    if results is None:
        return image_rgb, landmarks

    face_entries = []
    if hasattr(results, "face_landmarks"):
        face_entries = results.face_landmarks
    elif hasattr(results, "multi_face_landmarks"):
        face_entries = results.multi_face_landmarks
    else:
        return image_rgb, landmarks

    if not face_entries:
        return image_rgb, landmarks

    first_face = face_entries[0]
    landmark_list = getattr(first_face, "landmark", first_face)

    for i in range(min(len(landmark_list), 478)):
        landmark = landmark_list[i]
        landmarks[i, 0] = landmark.x
        landmarks[i, 1] = 1.0 - landmark.y

    if image_rgb is None:
        return None, landmarks

    # Optional plotting support for the legacy drawing API only.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "drawing_utils"):
        try:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image_rgb,
                landmark_list=first_face,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp.solutions.drawing_utils.draw_landmarks(
                image=image_rgb,
                landmark_list=first_face,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
            )
            mp.solutions.drawing_utils.draw_landmarks(
                image=image_rgb,
                landmark_list=first_face,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
        except Exception:
            pass

    return image_rgb, landmarks

def normalize_L0(landmarks):
    """
    Translate landmarks so Landmark 0 (upper lip central anchor) is placed at the origin (0, 0).
    Provides translation/position invariance within the camera frame.
    Accepts either a 1D flat array of shape (956,) or a 2D array of shape (478, 2).
    """
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.all(arr == 0):
        return arr.copy()

    was_flat = (arr.ndim == 1)
    if was_flat:
        if arr.shape[0] != 956:
            return arr.copy()
        pts = arr.reshape(-1, 2).copy()
    elif arr.ndim == 2 and arr.shape == (478, 2):
        pts = arr.copy()
    else:
        return arr.copy()

    pts -= pts[0].copy()
    return pts.reshape(-1) if was_flat else pts


def normalize_all_samples_L0(landmarks):
    """Applies L0 translation normalization to a batch of landmark samples."""
    normalized_landmarks = np.copy(landmarks)
    for i in range(landmarks.shape[0]):
        normalized_landmarks[i] = normalize_L0(landmarks[i])
    return normalized_landmarks


def normalize_size(landmarks, target_dist=0.25):
    """
    Scale landmarks relative to Landmark 0 so that the Euclidean distance between
    Landmark 0 and the facial centroid equals `target_dist`.
    Provides invariance to face size and distance to the camera.
    Accepts either a 1D flat array of shape (956,) or a 2D array of shape (478, 2).
    """
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.all(arr == 0):
        return arr.copy()

    was_flat = (arr.ndim == 1)
    if was_flat:
        if arr.shape[0] != 956:
            return arr.copy()
        pts = arr.reshape(-1, 2).copy()
    elif arr.ndim == 2 and arr.shape == (478, 2):
        pts = arr.copy()
    else:
        return arr.copy()

    anchor = pts[0].copy()
    centroid = np.mean(pts, axis=0, dtype=np.float32)
    distance = float(np.linalg.norm(centroid - anchor))

    if not np.isfinite(distance) or distance <= np.finfo(np.float32).eps:
        return arr.copy()

    scale = float(target_dist) / distance
    pts = anchor + (pts - anchor) * scale
    return pts.reshape(-1) if was_flat else pts


def normalize_all_samples_size(landmarks, target_dist=0.25):
    """Applies size/scale normalization to a batch of landmark samples."""
    normalized_landmarks = np.copy(landmarks)
    for i in range(landmarks.shape[0]):
        normalized_landmarks[i] = normalize_size(landmarks[i], target_dist)
    return normalized_landmarks


def normalize_rotation(landmarks, left_eye_idx=33, right_eye_idx=263):
    """
    Rotate landmarks in-plane around Landmark 0 so that the inter-ocular line
    connecting the outer eye corners (MediaPipe landmarks 33 and 263) is horizontal (0 deg roll).
    Provides invariance to in-plane face rotation / head tilt.
    Accepts either a 1D flat array of shape (956,) or a 2D array of shape (478, 2).
    """
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.all(arr == 0):
        return arr.copy()

    was_flat = (arr.ndim == 1)
    if was_flat:
        if arr.shape[0] != 956:
            return arr.copy()
        pts = arr.reshape(-1, 2).copy()
    elif arr.ndim == 2 and arr.shape == (478, 2):
        pts = arr.copy()
    else:
        return arr.copy()

    dx = float(pts[right_eye_idx, 0] - pts[left_eye_idx, 0])
    dy = float(pts[right_eye_idx, 1] - pts[left_eye_idx, 1])
    if abs(dx) <= np.finfo(np.float32).eps and abs(dy) <= np.finfo(np.float32).eps:
        return arr.copy()

    angle = math.atan2(dy, dx)
    cos_a = float(math.cos(-angle))
    sin_a = float(math.sin(-angle))

    anchor = pts[0].copy()
    shifted = pts - anchor
    rot_x = shifted[:, 0] * cos_a - shifted[:, 1] * sin_a
    rot_y = shifted[:, 0] * sin_a + shifted[:, 1] * cos_a
    pts[:, 0] = rot_x + anchor[0]
    pts[:, 1] = rot_y + anchor[1]

    return pts.reshape(-1) if was_flat else pts


def normalize_all_samples_rotation(landmarks, left_eye_idx=33, right_eye_idx=263):
    """Applies in-plane rotation normalization to a batch of landmark samples."""
    normalized_landmarks = np.copy(landmarks)
    for i in range(landmarks.shape[0]):
        normalized_landmarks[i] = normalize_rotation(
            landmarks[i], left_eye_idx=left_eye_idx, right_eye_idx=right_eye_idx
        )
    return normalized_landmarks


def normalize_landmarks_pipeline(landmarks, normalizations=("L0", "size", "rotation"), target_dist=0.25):
    """
    Applies a configurable sequence of geometric normalizations ('L0', 'size', 'rotation')
    to a single face's landmarks ((956,) or (478, 2)).
    """
    out = np.asarray(landmarks, dtype=np.float32).copy()
    if not normalizations:
        return out
    norm_set = [str(n).lower() for n in normalizations]
    if "l0" in norm_set:
        out = normalize_L0(out)
    if "size" in norm_set:
        out = normalize_size(out, target_dist=target_dist)
    if "rotation" in norm_set or "rot" in norm_set:
        out = normalize_rotation(out)
    return out


def normalize_all_samples(landmarks, normalizations=("L0", "size", "rotation"), target_dist=0.25):
    """
    Applies a configurable sequence of geometric normalizations ('L0', 'size', 'rotation')
    to an entire dataset array of shape (N, 956) or (N, 478, 2).
    """
    out = np.asarray(landmarks, dtype=np.float32).copy()
    if not normalizations:
        return out
    norm_set = [str(n).lower() for n in normalizations]
    if "l0" in norm_set:
        out = normalize_all_samples_L0(out)
    if "size" in norm_set:
        out = normalize_all_samples_size(out, target_dist=target_dist)
    if "rotation" in norm_set or "rot" in norm_set:
        out = normalize_all_samples_rotation(out)
    return out


def plot_normalization_stages(landmarks, target_dist=0.25, left_eye_idx=33, right_eye_idx=263):
    """
    Plots a 4-panel visual comparison showing the geometric effect of each normalization stage:
      1. Raw Landmarks (original position, scale, and tilt)
      2. After L0 (translation to origin)
      3. After L0 + size (camera-distance / scale normalization)
      4. After L0 + size + rotation (in-plane roll alignment via eye corners 33 & 263)
    """
    raw = np.asarray(landmarks, dtype=np.float32).reshape(478, 2)
    s_l0 = normalize_L0(raw)
    s_size = normalize_size(s_l0, target_dist=target_dist)
    s_rot = normalize_rotation(s_size, left_eye_idx=left_eye_idx, right_eye_idx=right_eye_idx)

    stages = [
        ("1. Raw Landmarks", raw),
        ("2. + L0 (Translation)", s_l0),
        (f"3. + Size (d={target_dist:.2f})", s_size),
        ("4. + Rotation (0° Eye-Line)", s_rot),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (title, pts) in zip(axes, stages):
        centroid = np.mean(pts, axis=0)
        dist = float(np.linalg.norm(centroid - pts[0]))
        dx = float(pts[right_eye_idx, 0] - pts[left_eye_idx, 0])
        dy = float(pts[right_eye_idx, 1] - pts[left_eye_idx, 1])
        angle_deg = math.degrees(math.atan2(dy, dx))

        ax.scatter(pts[:, 0], pts[:, 1], s=6, c="#1f77b4", alpha=0.75, label="Face Mesh (478)")
        ax.scatter(pts[0, 0], pts[0, 1], s=55, c="#d62728", marker="o", zorder=5, label="L0 (Origin Anchor)")
        ax.scatter(centroid[0], centroid[1], s=55, c="#2ca02c", marker="X", zorder=5, label="Centroid")
        # Inter-ocular line (33 -> 263)
        ax.plot(
            [pts[left_eye_idx, 0], pts[right_eye_idx, 0]],
            [pts[left_eye_idx, 1], pts[right_eye_idx, 1]],
            color="#ff7f0e",
            linewidth=2.2,
            zorder=4,
            label="Inter-ocular axis",
        )
        ax.set_title(f"{title}\nd={dist:.3f} | tilt={angle_deg:+.1f}°", fontsize=10.5, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="lower left", fontsize=8)
    plt.suptitle("Geometric Face Landmark Normalization Pipeline (L0 → Size → Rotation)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.show()


def perturb_landmarks_batch(
    landmarks_batch,
    scale_range=(0.55, 1.65),
    shift_range=(-0.25, 0.25),
    angle_deg_range=(-35.0, 35.0),
    seed=42,
):
    """
    Simulates real-world camera variations on a batch of raw landmarks (N, 956):
      - Different face distances to the camera / face sizes (random scaling)
      - Different head tilt angles (random in-plane rotation in degrees)
      - Different face positions in the frame (random 2D translation)
    Used in the notebook ablation study to demonstrate why raw/partially-normalized
    models fail under realistic camera conditions while full normalization remains invariant.
    """
    rng = np.random.default_rng(seed)
    out = np.asarray(landmarks_batch, dtype=np.float32).copy()
    for i in range(out.shape[0]):
        pts = out[i].reshape(478, 2)
        if np.all(pts == 0):
            continue
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid

        # 1. Random scale (simulating closer/farther distance from camera)
        scale = float(rng.uniform(scale_range[0], scale_range[1]))
        scaled = centered * scale

        # 2. Random head tilt rotation
        angle_rad = math.radians(float(rng.uniform(angle_deg_range[0], angle_deg_range[1])))
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rot_x = scaled[:, 0] * cos_a - scaled[:, 1] * sin_a
        rot_y = scaled[:, 0] * sin_a + scaled[:, 1] * cos_a
        rotated = np.column_stack((rot_x, rot_y))

        # 3. Random translation in camera frame
        shift = rng.uniform(shift_range[0], shift_range[1], size=(1, 2)).astype(np.float32)
        out[i] = (rotated + centroid + shift).reshape(-1)
    return out
    
def plot_face_landmarks(landmarks, ax):
    ax.clear()
    for i in range(len(landmarks)):
        plt.plot(landmarks[i, 0], landmarks[i, 1], "bo")

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = getattr(detection_result, "face_landmarks", None)
  if face_landmarks_list is None and hasattr(detection_result, "multi_face_landmarks"):
      face_landmarks_list = detection_result.multi_face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  if face_landmarks_list is None:
      return annotated_image

  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image


def FlattenFaceLandmarks(landmark_values):
    """Flatten a list of faces into one single x0,y0,x1,y1,... vector."""
    flat = []
    for face in landmark_values:
        for landmark in face:
            if isinstance(landmark, (list, tuple, np.ndarray)) and len(landmark) >= 2:
                flat.extend([float(landmark[0]), float(landmark[1])])
            else:
                flat.append(float(landmark))
    return flat


def GetFaceLandmarksListFromDetectionResult(detection_result):
    """Return a list of face landmark coordinates for all faces detected by MediaPipe."""
    landmark_values = []

    if detection_result is None:
        return landmark_values

    face_landmarks_list = getattr(detection_result, "face_landmarks", None)
    if face_landmarks_list is None and hasattr(detection_result, "multi_face_landmarks"):
        face_landmarks_list = detection_result.multi_face_landmarks

    if face_landmarks_list is None:
        return landmark_values

    for face_landmarks in face_landmarks_list:
        single_face = []
        for landmark in face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks:
            if hasattr(landmark, "x"):
                single_face.append([float(landmark.x), float(1.0 - landmark.y)])
            else:
                single_face.append([float(landmark[0]), float(landmark[1])])
        landmark_values.append(single_face)

    return landmark_values

# GetFaceLandmarksFromImages function
# The function takes a list of image file names,
# processes each image to detect face landmarks using MediaPipe Face Landmarker,
# extracts the X and Y coordinates of the landmarks,
# and creates a DataFrame containing this information along with additional metadata.
# The DataFrame is then saved to a CSV file, and the function handles exceptions and logs errors encountered during the process.
def GetFaceLandmarksFromImages(detector, IMAGE_FILES, images_path, images_subfolder, images_class, landmarks_path, annotations_path, config):
  
    columns = [["x" + str(i), "y" + str(i)] for i in range(config.num_landmarks)]  # [x0, y0],[x1,y1],[x2,y2]...
    columns = [item for sublist in columns for item in sublist]  # ['x0', 'y0', 'x1', 'y1' ....
    df_columns = columns + ["label", "frame", "path"]

    if not os.path.exists(images_path):
        # If the folder where the images are stored does not exist, we abort and log the event
        # This is an error because we cannot process the images if the folder does not exist
        print('\t[ERROR!!!][%s][FOLDER DOES NOT EXIST!!! ABORTING...]' % images_path)
        with open('HAR_mediapipe/logs.txt', 'a') as f:
            print('GetLandmarksFromImages()', images_path, file=f)
        return None     

    out_landmarks_path = landmarks_path + '/' + images_subfolder + '/'
    
    if not os.path.exists(out_landmarks_path):
        print('\t[%s][Folder does not exist!!! We create it!]' % out_landmarks_path)
        os.makedirs(out_landmarks_path)
        
    if config.save_images:
        out_path_imgs = annotations_path + '/' + images_subfolder + '/' 
     
        if not os.path.exists(out_path_imgs):
            print('\t[%s][Folder does not exist!!! We create it!]' % out_path_imgs)
            os.makedirs(out_path_imgs)
        
        out_path_imgs = out_path_imgs + images_class + '/'
        
        if not os.path.exists(out_path_imgs):
            print('\t[%s][Folder does not exist!!! We create it!]' % out_path_imgs)
            os.makedirs(out_path_imgs)

    out_path_df = os.path.join(out_landmarks_path + '/' + images_subfolder + '_' + images_class + '_faces_landmarks.csv')
    print('\t[New .csv file with landmarks][%s]' % out_path_df)

    print('\n[GetLandmarksFromImages]')
    print('\t[files][%s]' % str(IMAGE_FILES))
    print('\t[input images in path][%s]' % images_path)
    print('\t[landmarks out in path][%s]' % landmarks_path)
    print('\t[annotations out in path][%s]' % annotations_path)
    print('\t[columns][%s]' % str(df_columns))
    print('\n')

    successful_detections_df = pd.DataFrame([], columns=df_columns)

    num_successful_detections = 0
    num_failed_detections = 0

    for idx, file in enumerate(IMAGE_FILES):
        path_img = images_path + '/' + images_class + '/' + file
        
        if not os.path.exists(path_img):
          print(f"File does not exist: {path_img}")
          return None

        # Load the input image.
        image = cv2.imread(path_img)
        
        if image is None:
          print(f"Error: image could not be loaded: {path_img}")
          with open('logs.txt', 'a') as f:
            print('extract_XYZ()', images_path, idx + 1, file, file=f)
          return None
        
        #image_height, image_width, _ = image.shape

        # Convert the BGR image to RGB and processes each image to detect hand landmarks using MediaPipe Hands
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
            num_failed_detections = num_failed_detections + 1

            print(f'\t[ERROR!!!][{num_failed_detections}][{path_img}][FAILED TO DETECT FACE!!]')
            with open('HAR_mediapipe/logs.txt', 'a') as f:
                print('GetFaceLandmarksFromImages()', images_path, idx + 1, file, file=f)
        else:
            landmark_values = GetFaceLandmarksListFromDetectionResult(detection_result)

            # If landmark_values is empty we iterate and ignore current sample
            if not landmark_values:
                num_failed_detections = num_failed_detections + 1

                print(f'\t[ERROR!!!][{num_failed_detections}][{path_img}][FAILED TO DETECT FACE!!]')
                with open('HAR_mediapipe/logs.txt', 'a') as f:
                    print('GetFaceLandmarksFromImages()', images_path, idx + 1, file, file=f)
                continue

            # We have successfully detected a face in the image
            num_successful_detections = num_successful_detections + 1
            print(f'\t[{num_successful_detections}][{path_img}][SUCCESS!!]')

            if config.save_images:
                image = draw_landmarks_on_image(image, detection_result)
                out_annotated_image_path = out_path_imgs + file.split(".")[0] + '.jpg'
                # We save the annotated images to a specified output directory
                cv2.imwrite(out_annotated_image_path, image)
                print(f'\t[{out_annotated_image_path}][ANNOTATED IMAGE SAVED!!]')
            
            # Only samples with a successful detection result are stored.
            # Flatten to a single x0,y0,x1,y1,... vector before creating the DataFrame.
            df_landmark_values = FlattenFaceLandmarks(landmark_values)

            frame_number = file.split(".")[0]
            frame_number = frame_number.split("_")[1]

            # We add the image label, the frame number, and the image path to the DataFrame
            print(f"\timages_class = {images_class}, frame_number = {frame_number}, path_img = {path_img}")
            new_row = pd.DataFrame([df_landmark_values + [images_class, frame_number, path_img]], columns=df_columns)

            if successful_detections_df.empty:
                successful_detections_df = new_row
            else:
                successful_detections_df = pd.concat([successful_detections_df, new_row], ignore_index=True)

    # Finally we create a .csv file with the results of the landmarks detection
    successful_detections_df.to_csv(out_path_df, sep=",", header=True, index=False)

    return (successful_detections_df, num_successful_detections, num_failed_detections)
    # End of the GetLandmarksFromImages function

def remove_last_subfolder(path):
    head, tail = os.path.split(path)
    if tail:
        return head, tail
    elif head:
        new_head, new_tail = os.path.split(head)
        return new_head, new_tail
    else:
        return '', ''

def sorted_lists_match(list1, list2):
    sorted_list1 = sorted(list1)
    sorted_list2 = sorted(list2)

    return sorted_list1 == sorted_list2

def extract_classes_list_from_folders(folders_path, new_dataset_path):
  classes = []

  # Creation of a .txt file with different classes
  # First, we check the train folder substructure
  for root, dirs, files in os.walk(folders_path):
      for dir in dirs:
          classes.append(dir)

  classes = sorted(classes)

  root_path, processed_subfolder = remove_last_subfolder(folders_path)

  new_txt_filename = new_dataset_path + '/' + processed_subfolder + '_classes_list.txt'
  print('\n[extract_classes_list_from_folders][NEW .txt file]\n\t[%s]' % new_txt_filename)

  with open(new_txt_filename, 'w') as f:
    for c in classes:
      if c !='Landmarks' and c !='to_use'and c !='models':
        f.write(c + "\n")
    f.close()

  return classes

def load_individual_class_features_and_create_labeled_csv_dataset(input_mode, new_dataset_path, landmarks_path):
  classes_list_filename = new_dataset_path + '/' + input_mode + '_classes_list.txt'

  # Load file with classes
  print('\n[load_individual_class_features_and_create_labeled_csv_dataset][%s]\n\t[LOAD .txt file with the list of classes][%s]' % (input_mode, classes_list_filename))
  classes_list = pd.read_csv(classes_list_filename, header=None)
  classes_list = classes_list.rename(columns={0: "symbol"})
  classes_list["label"] = range(1, len(classes_list) + 1)

  # Load the first
  j = 0
  common_landmarks_folder = landmarks_path
  csv_filename = common_landmarks_folder + input_mode + '/' + input_mode + '_' + classes_list.loc[j]['symbol'] + '_faces_landmarks.csv'
  print('\t[LOAD .csv file with landmarks][%s]' % csv_filename)
  df = pd.read_csv(csv_filename)
  df['label'] = np.ones(len(df)) * classes_list.loc[j]["label"]

  # Load the rest
  for j in range(1, len(classes_list)):
    csv_filename = common_landmarks_folder + input_mode + '/' + input_mode + '_' + classes_list.loc[j]['symbol'] + '_faces_landmarks.csv'
    print('\t[LOAD .csv file with landmarks][%s]' % csv_filename)
    current_df = pd.read_csv(csv_filename)
    current_df['label'] = np.ones(len(current_df)) * classes_list.loc[j]['label']
    df = pd.concat([df, current_df])

  if not os.path.exists(new_dataset_path):
      # If the folder where the new dataset will be saved does not exist, this is an major error because we cannot save the new dataset, so we abort and log the event
      print('\t[ERROR!!!][%s][FOLDER DOES NOT EXIST!!! ABORTING...]' % new_dataset_path)
      with open('HAR_mediapipe/logs.txt', 'a') as f:
          print('load_individual_class_features_and_create_labeled_csv_dataset()', new_dataset_path, file=f)
      return None
  
  new_csv_filename = new_dataset_path + '/' + input_mode + '_dataset_with_labels.csv'
  print('\t[CREATING NEW .csv file][%s]' % new_csv_filename)
  df.to_csv(new_csv_filename, index=False)

# Create file CSV and Numpy
def create_numpy_with_feats_and_csv_with_just_labels(input_mode, new_dataset_path, NO_EMOTION=False):
  if NO_EMOTION:
    aux = '_plus_NO_EMOTION'
  else:
    aux = ''

  print('\n[create_numpy_with_feats_and_csv_with_just_labels][%s]' % input_mode)
  input_file = new_dataset_path + '/' + input_mode + '_dataset_with_labels' + aux + '.csv'
  print('\t[input_file][%s]' % input_file)
  df = pd.read_csv(input_file)
  num_points = 956 # Number of facial landmarks

  new_df = pd.DataFrame([], columns=["frame", "label"])
  data = np.zeros((len(df), num_points))
  fea_list = [["x" + str(j), "y" + str(j)] for j in range(int(num_points / 2))]
  flat_fea_list = [item for sublist in fea_list for item in sublist]

  for i in range(len(df)):
      data[i] = df.loc[i][flat_fea_list]
      new_df.loc[i] = [df["frame"][i], df["label"][i]]

  # Save NEW .csv which includes just the labels information
  new_csv_filename = new_dataset_path + '/' + input_mode + '_labels' + aux + '.csv'
  print('\t[NEW .csv][%s]' % new_csv_filename)
  new_df.to_csv(new_csv_filename, index=False)

  # Save NEW .npy which includes only the features
  new_npy_filename = new_dataset_path + '/' + input_mode + '_dataset' + aux + '.npy'
  print('\t[NEW .npy][%s]' % new_npy_filename)
  np.save(new_npy_filename, data)

def ArrangeInputDataForNetwork (x_data, debug = False):
  # x_data is expected as (num_samples, num_features_flat), where features are interleaved as x0,y0,x1,y1,...
  if debug:
    print('x_data')
    print(x_data.shape)
    print(x_data)

  # Allocate (num_samples, num_landmarks, 2) to separate x/y coordinates per landmark.
  x_data_reshaped = np.zeros((x_data.shape[0],
                              int(x_data.shape[1] / 2),
                              2))
  if debug:
    print('x_data_reshaped')
    print(x_data_reshaped.shape)
    print(x_data_reshaped)

  # Split even columns into x channel and odd columns into y channel.
  for i in range(len(x_data)):
    x_data_reshaped[i, :, 0] = x_data[i, 0::2]
    x_data_reshaped[i, :, 1] = x_data[i, 1::2]

  # Add a final singleton channel dimension for CNN-style input: (N, landmarks, 2, 1).
  x_data_out = np.reshape(x_data_reshaped, (x_data_reshaped.shape[0], x_data_reshaped.shape[1], x_data_reshaped.shape[2], 1))

  if debug:
    print('x_data_out')
    print(x_data_out.shape)
    print(x_data_out)

  # Return tensor ready for network input.
  return x_data_out


def run_normalization_ablation_study(
    raw_train_data,
    y_train_onehot,
    raw_test_data,
    y_test_onehot,
    define_network_model_fn,
    num_classes,
    network_type="CNN",
    num_cnn_features=64,
    dropout=0.3,
    batch_size=8,
    epochs=80,
    patience=20,
    learning_rate=1e-4,
    configs=None,
):
    """
    Runs a systematic ablation study across landmark normalization configurations:
      1. None: []
      2. Position only: ['L0']
      3. Position + Scale/Distance: ['L0', 'size']
      4. Position + Rotation: ['L0', 'rotation']
      5. Full (Position + Scale/Distance + Rotation): ['L0', 'size', 'rotation']

    Evaluates each trained model both on:
      - The original Test Set ('Standard Test Acc (%)')
      - A Geometrically Perturbed Test Set simulating real-time camera variations:
        changes in user distance/face size, camera frame shifts, and head tilt ('Perturbed Test Acc (%)').
    """
    import time
    import keras
    from keras.optimizers import AdamW
    from keras.callbacks import EarlyStopping
    from sklearn.metrics import accuracy_score, f1_score

    if configs is None:
        configs = [
            ("1. No Normalization ([])", []),
            ("2. Translation Only (['L0'])", ["L0"]),
            ("3. Translation + Size (['L0', 'size'])", ["L0", "size"]),
            ("4. Translation + Rotation (['L0', 'rotation'])", ["L0", "rotation"]),
            ("5. Full: L0 + Size + Rotation", ["L0", "size", "rotation"]),
        ]

    # Create a geometrically perturbed version of raw_test_data (simulating real camera movement/tilt/distance)
    perturbed_raw_test = perturb_landmarks_batch(
        raw_test_data,
        scale_range=(0.55, 1.65),
        shift_range=(-0.25, 0.25),
        angle_deg_range=(-35.0, 35.0),
        seed=42,
    )
    y_true_test = np.argmax(y_test_onehot, axis=1)

    ablation_records = []

    for label, norm_list in configs:
        print(f"\n{'=' * 75}")
        print(f"[ABLATION EXPERIMENT] {label} | normalizations = {norm_list}")
        print(f"{'=' * 75}")

        np.random.seed(2022)
        keras.utils.set_random_seed(2022)

        x_tr_norm = normalize_all_samples(raw_train_data, normalizations=norm_list)
        x_te_norm = normalize_all_samples(raw_test_data, normalizations=norm_list)
        x_te_pert_norm = normalize_all_samples(perturbed_raw_test, normalizations=norm_list)

        x_tr = ArrangeInputDataForNetwork(x_tr_norm)
        x_te = ArrangeInputDataForNetwork(x_te_norm)
        x_te_pert = ArrangeInputDataForNetwork(x_te_pert_norm)

        model = define_network_model_fn(
            num_classes=num_classes,
            network_type=network_type,
            num_cnn_features=num_cnn_features,
            dropout=dropout,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )
        opt = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        t0 = time.time()
        model.fit(
            x_tr,
            y_train_onehot,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            shuffle=True,
            validation_data=(x_te, y_test_onehot),
            callbacks=[early_stopping],
        )
        elapsed = time.time() - t0

        preds_std = np.argmax(model.predict(x_te, batch_size=batch_size, verbose=0), axis=1)
        preds_pert = np.argmax(model.predict(x_te_pert, batch_size=batch_size, verbose=0), axis=1)

        acc_std = 100.0 * accuracy_score(y_true_test, preds_std)
        f1_std = 100.0 * f1_score(y_true_test, preds_std, average="macro")
        acc_pert = 100.0 * accuracy_score(y_true_test, preds_pert)
        f1_pert = 100.0 * f1_score(y_true_test, preds_pert, average="macro")

        print(
            f"  -> Standard Test Acc : {acc_std:6.2f}% (Macro-F1: {f1_std:6.2f}%)\n"
            f"  -> Perturbed Test Acc: {acc_pert:6.2f}% (Macro-F1: {f1_pert:6.2f}%) "
            f"[Simulated distance/shift/rotation variations]"
        )

        ablation_records.append({
            "Configuration": label,
            "Normalizations": str(norm_list),
            "Standard Test Acc (%)": round(acc_std, 2),
            "Standard Macro-F1 (%)": round(f1_std, 2),
            "Perturbed Test Acc (%)": round(acc_pert, 2),
            "Perturbed Macro-F1 (%)": round(f1_pert, 2),
            "Training Time (s)": round(elapsed, 2),
        })

    df_ablation = pd.DataFrame(ablation_records)
    print("\n================ ABLATION STUDY SUMMARY ================")
    print(df_ablation.to_string(index=False))

    # Plot side-by-side comparison bar chart
    x_idx = np.arange(len(df_ablation))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.2))
    b1 = ax.bar(
        x_idx - width / 2,
        df_ablation["Standard Test Acc (%)"],
        width,
        label="Standard Test Set",
        color="#1f77b4",
        edgecolor="black",
    )
    b2 = ax.bar(
        x_idx + width / 2,
        df_ablation["Perturbed Test Acc (%)"],
        width,
        label="Perturbed Test Set (Distance + Shift + Rotation variations)",
        color="#ff7f0e",
        edgecolor="black",
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels(df_ablation["Configuration"], rotation=15, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Ablation Study: Impact of Landmark Normalizations on Emotion Recognition", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(h + 1.5, 105.0),
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()

    return df_ablation
