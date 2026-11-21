# -*- coding: utf-8 -*-
"""
Recorder configuration for HAR_INERCIAL_v2
@author: ffm
"""
import os

RASPI = 1
debug = 0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Number of recorded instances (4-second windows) per activity
TRAINING_LENGTH_PER_ACTIVITY = 30

# Directory where recorded CSV datasets are saved
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "my_dataset") + "/"

# Number of users to be recorded
NUM_USERS = 1

# Name or identifier for each specific action to be recorded separated by '_'
# Activities are recorded in the same order they are specified
DATA_SET = "DRIVE_BACKHAND_SERVE_LOB"
# DATA_SET = "SHAKE_CIRCLES_TILT"
# DATA_SET = "UP_SIT_PITCH_ROLL_YAW"
# DATA_SET = "RUNNING_STEP_SQUAT_CIRCLES"
