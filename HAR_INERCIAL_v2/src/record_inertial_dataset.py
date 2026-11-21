#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inertial Dataset Recorder for Raspberry Pi + SenseHAT (HAR_INERCIAL_v2)
Records 3-axis accelerometer windows (200 samples @ 50 Hz = 600 features + class label)
directly into CSV format ready for Google Colab training.
"""
import os
import sys
import tty
import termios
import threading
import traceback
import csv
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime

from AR_modules.config.general_recorder import RASPI, debug, TRAINING_LENGTH_PER_ACTIVITY
from AR_modules.config.general_recorder import DATA_PATH, DATA_SET, NUM_USERS
from AR_modules.timing.timing import RepeatedTimer, AccelerometerRecorder

if RASPI:
    from AR_modules.sensehat.RTIMULib import InitializeRTIMU
    from AR_modules.sensehat.displayLib import ShowInitialMsg, InitializeDisplay
    from AR_modules.sensehat.displayLib import RECORDING, OFF, colour_array
    sys.path.append('.')
    import RTIMU

parser = argparse.ArgumentParser(description='Record 3-axis inertial activity dataset.')
parser.add_argument('-t', '-T', nargs=1, dest='train_length_per_activity',
                    help='Number of 4-second windows to record per activity')
parser.add_argument('-d', '-D', nargs='?', const=1, default=0, help='DEBUG MODE ON')
args = parser.parse_args()

print("\nargs:")
print(args)

if args.train_length_per_activity is not None and int(args.train_length_per_activity[0]) > 0:
    TRAINING_LENGTH_PER_ACTIVITY = int(args.train_length_per_activity[0])

debug = args.d
columns = defaultdict(list)

if RASPI:
    print("\n[InitializeRTIMU]")
    SETTINGS_FILE = "RTIMULib"
    print("Using settings file " + SETTINGS_FILE + ".ini")
    if not os.path.exists(SETTINGS_FILE + ".ini"):
        print("Settings file does not exist, will be created")

    s = RTIMU.Settings(SETTINGS_FILE)
    imu = RTIMU.RTIMU(s)
    print("IMU Name: " + imu.IMUName())

    if not imu.IMUInit():
        print("IMU Init Failed")
        sys.exit(1)
    else:
        print("IMU Init Succeeded")

    imu.setAccelEnable(True)
    poll_interval = imu.IMUGetPollInterval()
    print("Recommended Poll Interval: %dmS\n" % poll_interval)
else:
    imu = 0

recorder = AccelerometerRecorder(
    150,   # WINDOW_SIZE
    100,   # OVERLAP (mini_buffer_size = 50 -> buffer_size = 200 samples = 4s @ 50Hz)
    0.02,  # INTERVAL in secs == 20 ms (50 Hz)
    imu
)

print("\n[ACTIVITY RECORDER - HAR_INERCIAL_v2]\n")
print("\t[DEBUG MODE][%d]\n" % debug)
recorder.printRecorderInfo()


def wait_for_keypress(target_key, exit_key, yes_to_all_key):
    print(f"Press '{target_key}' to continue or '{exit_key}' to exit... ('{yes_to_all_key}') if YES to all...")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    char = ''

    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            if char == target_key:
                break
            elif char == yes_to_all_key:
                print("Yes to all...")
                break
            elif char == exit_key:
                print("Exiting...")
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char


def record_raw_dataset(csv_path, num_frames_to_record=TRAINING_LENGTH_PER_ACTIVITY):
    successful_recording = True
    print("\n[RECORDING NEW DATA]")
    print(f"\t[recording {num_frames_to_record} new frames per class]\n")

    activities = DATA_SET.split("_")
    num_activities = len(activities)

    global recorder
    print("RECORDING ACTIVITIES FROM", NUM_USERS, "USERS.")
    print("\t[ACTIVITY SET]")
    for step in range(num_activities):
        print(f"\t\t[{step + 1}][{activities[step]}]")

    csv_data = []

    for user in range(NUM_USERS):
        print(f"[USER][{user}]")
        print('[Change the user and get ready to start recording!]')

        for step in range(num_activities):
            recorder.ResetRecordingParameters()
            print(f'\n[ACTIVITY][{activities[step]}]')

            key = wait_for_keypress('c', 'q', 'y')
            if key == 'q':
                successful_recording = False
                break

            timer = RepeatedTimer(0.02, recorder.ReadAccelerometer)
            num_frames_recorded = 0

            print("RECORDING!!!!")
            if RASPI:
                sense.set_pixels(RECORDING)

            try:
                while num_frames_recorded < num_frames_to_record:
                    recorder.condition.acquire()
                    try:
                        while True:
                            new_data = None
                            recorder.count_lock.acquire()
                            try:
                                if recorder.item:
                                    # 600 accelerometer features (200 X + 200 Y + 200 Z) + nominal class label
                                    new_data = (
                                        list(recorder.AccXarray)
                                        + list(recorder.AccYarray)
                                        + list(recorder.AccZarray)
                                        + [activities[step]]
                                    )
                                    recorder.item = 0
                            finally:
                                recorder.count_lock.release()

                            if new_data is not None:
                                csv_data.append(new_data)
                                if debug:
                                    print(new_data)
                                num_frames_recorded += 1
                                print(
                                    f"\rRecorded {num_frames_recorded} out of {num_frames_to_record} frames.",
                                    end='',
                                    flush=True,
                                )
                                break

                            recorder.condition.wait()
                    finally:
                        recorder.condition.release()
            finally:
                timer.stop()
                if RASPI:
                    sense.set_pixels(OFF)

        if not successful_recording:
            break

    if successful_recording and csv_data:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        print(f'\n[RECORDED DATA SAVED IN {csv_path}]')
        with open(csv_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerows(csv_data)


if __name__ == "__main__":
    try:
        if RASPI:
            print("[InitializeDisplay]")
            sense = InitializeDisplay()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        os.makedirs(DATA_PATH, exist_ok=True)
        csv_path = os.path.join(DATA_PATH, f"{DATA_SET}_{timestamp}.csv")

        print(csv_path)
        record_raw_dataset(csv_path)

    except Exception:
        print(traceback.format_exc())

    finally:
        all_threads = list(threading.enumerate())
        print('Number of existing threads = %d' % len(all_threads))
        print("\nnum_mini_buffers_recorded = " + str(recorder.num_mini_buffers_recorded))
        print("num_samples_recorded = " + str(recorder.num_samples_recorded))
        sys.exit(0)
