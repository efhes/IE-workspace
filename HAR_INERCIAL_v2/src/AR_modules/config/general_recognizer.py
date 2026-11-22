# -*- coding: utf-8 -*-
"""
Recognizer configuration for HAR_INERCIAL_v2
@author: ffm
"""
import os

RASPI = 1
debug = 0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Folder where trained models are stored
MODELS_PATH = os.path.join(PROJECT_ROOT, "models") + "/"

# Model name (root filename preceding .json/.weights.h5, .keras, or .npz files)
#MODEL_NAME = "CIRCLES_SHAKE_TILT-LSTM_from_raw"
MODEL_NAME = "BACKHAND_DRIVE_LOB_SERVE-LSTM_from_raw"

# Ordered list of defined classes separated by '_'
# Note: The notebook sorts classes alphabetically when one-hot encoding,
# and also exports a <MODEL_NAME>.classes.txt file that demo-lstm-recognizer.py
# reads automatically if present.
CLASSES = "CIRCLES_SHAKE_TILT"
# CLASSES = "BACKHAND_DRIVE_LOB_SERVE"
# CLASSES = "UP_SIT_PITCH_ROLL_YAW"
# CLASSES = "RUNNING_STEP_SQUAT_CIRCLES"
