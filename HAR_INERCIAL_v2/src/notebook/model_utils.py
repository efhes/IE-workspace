# -*- coding: utf-8 -*-
"""
Sequential Deep Learning (Keras 3.15.1) model definition, training, evaluation,
and Raspberry Pi export utilities for HAR_INERCIAL_v2.
"""
import os
import time
import numpy as np
import pandas as pd
import keras
from keras import Sequential
from keras.models import model_from_json, load_model
from keras.layers import (
    Input,
    LSTM,
    GRU,
    SimpleRNN,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    BatchNormalization,
)
from keras.optimizers import Adam, RMSprop, AdamW, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


def build_optimizer(optimizer_name="adam", learning_rate=1e-3, clipnorm=1.0):
    """
    Creates a Keras 3 optimizer with gradient norm clipping (clipnorm=1.0)
    to prevent exploding gradients (loss: nan) during Backpropagation Through Time (BPTT)
    over 200-step raw accelerometer sequences.
    """
    if not isinstance(optimizer_name, str):
        return optimizer_name

    name = optimizer_name.lower().strip()
    if name == "adam":
        return Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    elif name == "rmsprop":
        return RMSprop(learning_rate=learning_rate, clipnorm=clipnorm)
    elif name == "adamw":
        return AdamW(learning_rate=learning_rate, clipnorm=clipnorm)
    elif name == "sgd":
        return SGD(learning_rate=learning_rate, momentum=0.9, clipnorm=clipnorm)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Use 'adam', 'rmsprop', 'adamw', or 'sgd'.")


def CreateLSTM(
    neurons_first_layer=128,
    n_steps=200,
    dimension=3,
    num_classes=3,
    network_type="LSTM",
    dropout=0.2,
    learning_rate=1e-3,
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=("accuracy",),
):
    """
    Builds and compiles a Keras 3.15.1 Sequential model for 3D inertial time series (n_steps, dimension).
    Supported `network_type` options:
      - 'LSTM'        : Single unidirectional LSTM layer (compatible with Raspberry Pi Fast-NumPy engine!)
      - 'GRU'         : Gated Recurrent Unit layer
      - 'BiLSTM'      : Bidirectional LSTM layer
      - 'Conv1D_LSTM' : 1D Convolutional front-end + MaxPooling1D + LSTM layer
      - 'SimpleRNN'   : Vanilla Elman Recurrent Neural Network (for pedagogical comparison)
    """
    model = Sequential(name=f"HAR_{network_type}")
    model.add(Input(shape=(n_steps, dimension)))

    net = network_type.upper().strip()
    if net == "LSTM":
        model.add(LSTM(neurons_first_layer, return_sequences=False, activation="tanh"))
    elif net == "GRU":
        model.add(GRU(neurons_first_layer, return_sequences=False, activation="tanh"))
    elif net == "BILSTM":
        model.add(Bidirectional(LSTM(neurons_first_layer, return_sequences=False, activation="tanh")))
    elif net == "CONV1D_LSTM":
        model.add(Conv1D(filters=64, kernel_size=5, activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(neurons_first_layer, return_sequences=False, activation="tanh"))
    elif net == "SIMPLERNN":
        model.add(SimpleRNN(neurons_first_layer, return_sequences=False, activation="tanh"))
    else:
        raise ValueError(
            f"Unknown network_type '{network_type}'. Choose from: 'LSTM', 'GRU', 'BiLSTM', 'Conv1D_LSTM', 'SimpleRNN'."
        )

    if dropout and dropout > 0.0:
        model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation="softmax"))

    opt = build_optimizer(optimizer, learning_rate=learning_rate, clipnorm=1.0)
    model.compile(loss=loss, optimizer=opt, metrics=list(metrics))
    return model


def train_sequential_model(
    model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    batch_size=32,
    epochs=50,
    patience=10,
    validation_split=0.15,
    verbose=1,
):
    """
    Trains the sequential model with EarlyStopping and ReduceLROnPlateau callbacks.
    """
    monitor_metric = "val_loss"
    callbacks = [
        EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=max(3, patience // 3),
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    fit_kwargs = {
        "batch_size": batch_size,
        "epochs": epochs,
        "callbacks": callbacks,
        "verbose": verbose,
    }
    if X_test is not None and y_test is not None:
        fit_kwargs["validation_data"] = (X_test, y_test)
    elif validation_split and validation_split > 0.0 and len(X_train) >= 20:
        fit_kwargs["validation_split"] = validation_split
    else:
        fit_kwargs["callbacks"] = []

    t0 = time.perf_counter()
    history = model.fit(X_train, y_train, **fit_kwargs)
    train_time = time.perf_counter() - t0
    print(f"\n[Training Completed in {train_time:.2f} s]")
    return history, train_time


def EvalLSTM(y_labels, y_pred, num_classes, class_names=None):
    """
    Evaluates sequential model predictions against one-hot ground-truth labels.
    Prints confusion matrix, classification report, and accuracy percentage.
    """
    y_true_idx = np.argmax(y_labels, axis=1)
    y_pred_idx = np.asarray(y_pred).reshape(-1)

    target_names = [str(c) for c in class_names] if class_names is not None else None
    print(confusion_matrix(y_true_idx, y_pred_idx))
    print(classification_report(y_true_idx, y_pred_idx, target_names=target_names, digits=4))
    eval_accuracy = 100.0 * accuracy_score(y_true_idx, y_pred_idx)
    print("Accuracy %0.4f%%" % eval_accuracy)
    return float(eval_accuracy)


def Recognize_LSTM(trained_LSTM, test_parameters, test_labels, class_mapping, max_rows=30):
    """
    Runs inference on test_parameters and displays sample-by-sample predictions and probability distributions.
    """
    y_pred = trained_LSTM.predict(test_parameters, verbose=0)
    total = len(test_labels)
    limit = min(total, max_rows) if max_rows else total
    print(f"# - actual - predicted - error - class distribution (showing {limit} of {total} samples)")
    for index in range(limit):
        pred_idx = int(np.argmax(y_pred[index], axis=-1)) + 1
        true_idx = int(np.argmax(test_labels[index], axis=-1)) + 1
        pred = class_mapping.get(pred_idx, "UNK")
        true_val = class_mapping.get(true_idx, "UNK")
        probs_str = np.array2string(y_pred[index], precision=4, suppress_small=True)
        print(
            "%d - %s - %s - %s  - %s"
            % (index + 1, true_val, pred, "yes" if pred != true_val else "no", probs_str)
        )
    return y_pred


def save_model_for_raspi(model, models_dir, root_class_list, model_name="-LSTM_from_raw", classes_list=None):
    """
    Saves the trained Keras 3.15.1 model in both:
      1. Architecture (.json) + weights (.weights.h5) format for demo-lstm-recognizer.py
      2. Ordered class list (.classes.txt) so the Raspberry Pi recognizer maps class indices automatically.
    """
    os.makedirs(models_dir, exist_ok=True)
    base_path = os.path.join(models_dir, f"{root_class_list}{model_name}")
    json_path = base_path + ".json"
    weights_path = base_path + ".weights.h5"
    classes_path = base_path + ".classes.txt"

    with open(json_path, "w", encoding="utf-8") as json_file:
        json_file.write(model.to_json())

    model.save_weights(weights_path)

    if classes_list is not None:
        with open(classes_path, "w", encoding="utf-8") as cf:
            for cls_name in sorted(classes_list):
                cf.write(f"{cls_name}\n")

    print("[NEW MODEL][Successfully saved for Raspberry Pi!]")
    print(f"\tMODEL PREFIX : {root_class_list}{model_name}")
    print(f"\t.json file   : {json_path}")
    print(f"\t.weights.h5  : {weights_path}")
    if classes_list is not None:
        print(f"\t.classes.txt : {classes_path}")

    return json_path, weights_path


def load_saved_model(json_path, weights_path):
    """
    Loads a saved Keras 3 model from its .json architecture and .weights.h5 weights file.
    """
    with open(json_path, "r", encoding="utf-8") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    print(f"[Model successfully loaded from {json_path} and {weights_path}]")
    return loaded_model
