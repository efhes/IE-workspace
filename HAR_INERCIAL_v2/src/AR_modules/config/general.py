# -*- coding: utf-8 -*-
"""
General configuration for HAR_INERCIAL_v2
@author: ffm
"""
import os

RASPI = 1
debug = 0

# Resolve project root dynamically (fallback to /home/pi/workspace/HAR_INERCIAL_v2)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

TRAINING_LENGTH_PER_ACTIVITY = 30

CLASSIFIER_NAME = "BACKHAND_DRIVE_LOB_SERVE-LSTM_from_raw"

MODELS_PATH = os.path.join(PROJECT_ROOT, "models") + "/"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "my_dataset") + "/"

NUM_USERS = 1

# Activities recorded in the specified order (separated by '_')
DATA_SET = "DRIVE_BACKHAND_SERVE_LOB"
# DATA_SET = "SHAKE_CIRCLES_TILT"
# DATA_SET = "UP_SIT_PITCH_ROLL_YAW"
# DATA_SET = "RUNNING_STEP_SQUAT_CIRCLES"
