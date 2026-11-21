# -*- coding: utf-8 -*-
"""
High-precision 50 Hz accelerometer circular-buffer recorder for HAR_INERCIAL_v2
@author: ffm
"""

from AR_modules.config.general import RASPI, debug

import time
import threading
import numpy as np
from threading import Event, Thread


class AccelerometerRecorder:
    GRAVITY_ACCEL = 9.81

    def __init__(self, window_size, overlap, interval, imu, *args, **kwargs):
        self.window_size = window_size
        self.overlap = overlap
        self.mini_buffer_size = window_size - overlap
        self.interval = interval
        self.imu = imu

        self.count_lock = threading.Lock()
        self.condition = threading.Condition()

        if self.mini_buffer_size <= 0:
            print(f"\n[WRONG MINI-BUFFER SIZE ({self.mini_buffer_size})!!!]")
            print(f"[window_size = {self.window_size}]")
            print(f"[overlap = {self.overlap}]")
            raise ValueError("mini_buffer_size must be positive")

        if self.window_size % self.mini_buffer_size != 0:
            print(f"\n[WRONG MINI-BUFFER SIZE ({self.mini_buffer_size})!!!]")
            print(f"[REMAINDER OF WINDOW SIZE / mini_buffer_size MUST BE ZERO ({self.window_size} / {self.mini_buffer_size})]\n")
            raise ValueError("window_size must be a multiple of mini_buffer_size")

        self.num_mini_buffers_per_frame = int(self.window_size / self.mini_buffer_size) + 1
        self.num_mini_buffers = self.num_mini_buffers_per_frame + 30
        self.buffer_size = self.window_size + self.mini_buffer_size

        self.p_recording_buffer = 0
        self.p_item_in_recording_buffer = 0
        self.p_processing_buffer = 0

        self.num_mini_buffers_recorded = 0
        self.num_samples_recorded = 0
        self.count = 0
        self.item = 0

        self.AccX_bufferArray = np.zeros((self.num_mini_buffers, self.mini_buffer_size), dtype=np.float64)
        self.AccY_bufferArray = np.zeros((self.num_mini_buffers, self.mini_buffer_size), dtype=np.float64)
        self.AccZ_bufferArray = np.zeros((self.num_mini_buffers, self.mini_buffer_size), dtype=np.float64)

        self.AccXarray = np.zeros(self.buffer_size, dtype=np.float64)
        self.AccYarray = np.zeros(self.buffer_size, dtype=np.float64)
        self.AccZarray = np.zeros(self.buffer_size, dtype=np.float64)

    def printRecorderInfo(self):
        print(f"\t[GRAVITY_ACCEL = {self.GRAVITY_ACCEL}]")
        print(f"\t[window_size = {self.window_size}]")
        print(f"\t[overlap = {self.overlap}]")
        print(f"\t[mini_buffer_size = {self.mini_buffer_size}]")
        print(f"\t[buffer_size = {self.buffer_size} (num_mini_buffers_per_frame = {self.num_mini_buffers_per_frame})]")
        print(f"\t[num_mini_buffers = {self.num_mini_buffers}]\n")

    def ResetRecordingParameters(self):
        with self.count_lock:
            self.p_recording_buffer = 0
            self.p_item_in_recording_buffer = 0
            self.p_processing_buffer = 0
            self.num_mini_buffers_recorded = 0
            self.num_samples_recorded = 0
            self.item = 0

    def ReadAccelerometer(self):
        self.count_lock.acquire()
        try:
            self.count += 1
            self.num_samples_recorded += 1

            if RASPI and self.imu:
                self.imu.IMURead()
                accelerometer_data = self.imu.getAccel()

                x = float(accelerometer_data[0]) * self.GRAVITY_ACCEL
                y = float(accelerometer_data[1]) * self.GRAVITY_ACCEL
                z = float(accelerometer_data[2]) * self.GRAVITY_ACCEL

                if np.isnan(x) or np.isinf(x):
                    x = 0.0
                if np.isnan(y) or np.isinf(y):
                    y = 0.0
                if np.isnan(z) or np.isinf(z):
                    z = 0.0
            else:
                # Synthetic 3-axis test wave when running off-hardware (RASPI = 0)
                t = self.count * self.interval
                x = float(np.sin(2.0 * np.pi * 1.5 * t) * self.GRAVITY_ACCEL)
                y = float(np.cos(2.0 * np.pi * 1.5 * t) * self.GRAVITY_ACCEL)
                z = float(self.GRAVITY_ACCEL)

            self.AccX_bufferArray[self.p_recording_buffer, self.p_item_in_recording_buffer] = x
            self.AccY_bufferArray[self.p_recording_buffer, self.p_item_in_recording_buffer] = y
            self.AccZ_bufferArray[self.p_recording_buffer, self.p_item_in_recording_buffer] = z

            self.p_item_in_recording_buffer += 1

            if self.p_item_in_recording_buffer >= self.mini_buffer_size:
                self.num_mini_buffers_recorded += 1
                self.p_item_in_recording_buffer = 0
                self.p_recording_buffer = (self.p_recording_buffer + 1) % self.num_mini_buffers

                if self.p_recording_buffer == self.p_processing_buffer:
                    print("\n[ReadAccelerometer][WARNING: Circular buffer overrun - dropping oldest mini-buffer]\n")
                    self.p_processing_buffer = (self.p_processing_buffer + 1) % self.num_mini_buffers

                if self.num_samples_recorded >= self.buffer_size:
                    self.condition.acquire()
                    try:
                        self.item = 1
                        p_aux_item = 0
                        for value in range(self.num_mini_buffers_per_frame):
                            p_aux_buffer = (self.p_processing_buffer + value) % self.num_mini_buffers
                            end_idx = p_aux_item + self.mini_buffer_size
                            self.AccXarray[p_aux_item:end_idx] = self.AccX_bufferArray[p_aux_buffer, :]
                            self.AccYarray[p_aux_item:end_idx] = self.AccY_bufferArray[p_aux_buffer, :]
                            self.AccZarray[p_aux_item:end_idx] = self.AccZ_bufferArray[p_aux_buffer, :]
                            p_aux_item = end_idx

                        self.p_processing_buffer = (self.p_processing_buffer + 1) % self.num_mini_buffers
                        self.condition.notify()
                    finally:
                        self.condition.release()
        finally:
            self.count_lock.release()


class RepeatedTimer:
    """Repeat `function` every `interval` seconds without drift."""

    def __init__(self, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start = time.time()
        self.event = Event()
        self.thread = Thread(target=self._target, daemon=True)
        self.thread.start()

    def _target(self):
        while not self.event.wait(self._time):
            self.function(*self.args, **self.kwargs)

    @property
    def _time(self):
        return self.interval - ((time.time() - self.start) % self.interval)

    def stop(self):
        self.event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
