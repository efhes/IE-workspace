import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def get_XYZ(results, image_rgb):

    landmark_values = [[0, 0] for _ in range(21)]
    
    # esto para dibujar
    mp_drawing = mp.solutions.drawing_utils
    
    image_height, image_width, _ = image_rgb.shape
        
    for hand_landmarks in results.multi_hand_landmarks:
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
            landmark_values.append([landmark.x, landmark.y])

    return annotated_image, landmark_values

# GetLandmarksFromImages function
# The function takes a list of image file names,
# processes each image to detect hand landmarks using MediaPipe Hands,
# extracts the X and Y coordinates of the landmarks,
# and creates a DataFrame containing this information along with additional metadata.
# The DataFrame is then saved to a CSV file, and the function handles exceptions and logs errors encountered during the process.
def GetLandmarksFromImages(detector, IMAGE_FILES, images_path, images_subfolder, images_class, out_path, config):
  
    columns = [["x" + str(i), "y" + str(i)] for i in range(config.num_landmarks)]  # [x0, y0],[x1,y1],[x2,y2]...
    columns = [item for sublist in columns for item in sublist]  # ['x0', 'y0', 'x1', 'y1' ....
    df_columns = columns + ["label", "frame", "path"]

    print('\n[GetLandmarksFromImages]')
    print('\t[files][%s]' % str(IMAGE_FILES))
    print('\t[path][%s]' % images_path)
    print('\t[out_file][%s]' % out_path)
    print('\t[columns][%s]' % str(df_columns))
    print('\n')

    if not os.path.exists(out_path):
        print('\t[New folder][%s]' % out_path)
        os.makedirs(out_path)

    out_landmarks_path = out_path + '/landmarks/'
    if not os.path.exists(out_landmarks_path):
        print('\t[New folder][%s]' % out_landmarks_path)
        os.makedirs(out_landmarks_path)

    out_path_df = os.path.join(out_landmarks_path + '/' + images_subfolder + '_' + images_class + '_poses_landmarks.csv')
    print('\t[New file][%s]' % out_path_df)

    successful_detections_df = pd.DataFrame([], columns=df_columns)

    num_successful_detections = 0
    num_failed_detections = 0

    for idx, file in enumerate(IMAGE_FILES):
        path_img = images_path + '/' + images_subfolder + '/' + images_class + '/' + file
        # catch exception if the image is not found
        try:
            image = cv2.imread(path_img)
        
        except FileNotFoundError:
            print(f"Error: File not found at {path_img}.")
            with open('logs.txt', 'a') as f:
                print('extract_XYZ() ', images_path, ' ', idx + 1, ' ', file, file=f)
            return None
        except cv2.error as e: # Catch OpenCV specific errors.
            print(f"OpenCV Error: {e}")
            return None
        except Exception as e: # Catch any other potential exceptions.
            print(f"An unexpected error occurred: {e}")
            return None
        
        image_height, image_width, _ = image.shape

        # Convert the BGR image to RGB and processes each image to detect hand landmarks using MediaPipe Hands
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get hand landmarks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        landmark_values = []
        #if results.multi_hand_landmarks is None:
        if detection_result.hand_landmarks is None:
            # We have NOT successfully detected a hand in the image
            num_failed_detections = num_failed_detections + 1

            for i in range(config.num_landmarks):
                landmark_values += [0, 0]
            print('\t[ERROR!!!][NO LANDMARKS!!!][%s]' % path_img)
        else:
            # We have successfully detected a hand in the image
            num_successful_detections = num_successful_detections + 1
            
            # This function returns already normalized landmarks
            image, landmark_values = draw_landmarks_on_image(image, detection_result)

            if config.save_images:
                # We save the annotated images to a specified output directory
                cv2.imwrite(out_path_imgs + file.split(".")[0] + '.jpg', image)
            
            # Only samples with a successful detection result are stored
            # We first unwrap landmark_values to a arrange them in a single row before passing them to the DataFrame
            df_landmark_values = [item for sublist in landmark_values for item in sublist]
            
            frame_number = file.split(".")[0]
            frame_number = frame_number.split("_")[1]

            # We add the image label, the frame number, and the image path to the DataFrame
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

def load_individual_class_features_and_create_labeled_csv_dataset(input_mode, root_path, new_dataset_path):
  classes_list_filename = new_dataset_path + '/' + input_mode + '_classes_list.txt'

  # Load file with classes
  print('\n[load_individual_class_features_and_create_labeled_csv_dataset][%s]\n\t[LOAD .txt file with the list of classes][%s]' % (input_mode, classes_list_filename))
  classes_list = pd.read_csv(classes_list_filename, header=None)
  classes_list = classes_list.rename(columns={0: "symbol"})
  classes_list["label"] = range(1, len(classes_list) + 1)

  # Load the first
  j = 0
  common_landmarks_folder = new_dataset_path + '/landmarks/'
  csv_filename = common_landmarks_folder + input_mode + '_' + classes_list.loc[j]['symbol'] + '_poses_landmarks.csv'
  print('\t[LOAD .csv file with landmarks][%s]' % csv_filename)
  df = pd.read_csv(csv_filename)
  df['label'] = np.ones(len(df)) * classes_list.loc[j]["label"]

  # Load the rest
  for j in range(1, len(classes_list)):
    csv_filename = common_landmarks_folder + input_mode + '_' + classes_list.loc[j]['symbol'] + '_poses_landmarks.csv'
    print('\t[LOAD .csv file with landmarks][%s]' % csv_filename)
    current_df = pd.read_csv(csv_filename)
    current_df['label'] = np.ones(len(current_df)) * classes_list.loc[j]['label']
    df = pd.concat([df, current_df])

  if not os.path.exists(new_dataset_path):
      os.makedirs(new_dataset_path)
  new_csv_filename = new_dataset_path + '/' + input_mode + '_dataset_with_labels.csv'
  print('\t[CREATING NEW .csv file][%s]' % new_csv_filename)
  df.to_csv(new_csv_filename, index=False)

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
