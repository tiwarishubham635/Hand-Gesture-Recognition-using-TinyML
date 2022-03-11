from __future__ import absolute_import, division, print_function, unicode_literals
from cgitb import html

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import lite

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import load_model

# from tensorflow.keras.callbacks import TensorBoard
import numpy as np

import cv2
import numpy as np
import pandas as pd
import os
import time
import mediapipe as mp

DATA_PATH = os.path.join('ISL_Data')
actions = np.array(['A', 'B', 'C'])

# no of videos
no_sequences = 30

# no of frames in each video
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}
# print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(
                sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)

# Step 6 - preprocess data create labels and features

y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = load_model('action3.h5')

yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


def check():
    return accuracy_score(yhat, ytrue)


SAVED_MODEL = "saved_models"
# tf.saved_model.save(model, SAVED_MODEL)
sign_model = hub.load(SAVED_MODEL)
TFLITE_MODEL = "tflite_models/sign.tflite"
TFLITE_QUANT_MODEL = "tflite_models/sign_quant.tflite"

X_test = np.float32(X_test)


def convert_to_tflite():
    converter = lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    converted_tflite_model = converter.convert()
    open(TFLITE_MODEL, "wb").write(converted_tflite_model)

    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_quant_model = converter.convert()
    open(TFLITE_QUANT_MODEL, "wb").write(tflite_quant_model)

    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    tflite_interpreter.resize_tensor_input(
        input_details[0]['index'], X_test.shape)
    tflite_interpreter.resize_tensor_input(
        output_details[0]['index'], y_test.shape)
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    # Load quantized TFLite model
    tflite_interpreter_quant = tf.lite.Interpreter(
        model_path=TFLITE_QUANT_MODEL)

    # Learn about its input and output details
    input_details = tflite_interpreter_quant.get_input_details()
    output_details = tflite_interpreter_quant.get_output_details()

    # Resize input and output tensors
    tflite_interpreter_quant.resize_tensor_input(
        input_details[0]['index'], X_test.shape)
    tflite_interpreter_quant.resize_tensor_input(
        output_details[0]['index'], y_test.shape)
    tflite_interpreter_quant.allocate_tensors()

    input_details = tflite_interpreter_quant.get_input_details()
    output_details = tflite_interpreter_quant.get_output_details()

    # Run inference
    tflite_interpreter_quant.set_tensor(input_details[0]['index'], X_test)

    tflite_interpreter_quant.invoke()

    tflite_q_model_predictions = tflite_interpreter_quant.get_tensor(
        output_details[0]['index'])
#print("\nPrediction results shape:", tflite_q_model_predictions.shape)


def TestDataComparison():
    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    tflite_interpreter.resize_tensor_input(
        input_details[0]['index'], X_test.shape)
    tflite_interpreter.resize_tensor_input(
        output_details[0]['index'], y_test.shape)
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    tflite_interpreter.set_tensor(input_details[0]['index'], X_test)
    tflite_interpreter.invoke()
    tflite_model_predictions = tflite_interpreter.get_tensor(
        output_details[0]['index'])
    #print("Prediction results shape:", tflite_model_predictions.shape)

    # Convert prediction results to Pandas dataframe, for better visualization
    tflite_pred_dataframe = pd.DataFrame(tflite_model_predictions)
    tflite_pred_dataframe.columns = actions

    #print("TFLite prediction results: ")
    # print(tflite_pred_dataframe)

    tflite_interpreter_quant = tf.lite.Interpreter(
        model_path=TFLITE_QUANT_MODEL)

    input_details = tflite_interpreter_quant.get_input_details()
    output_details = tflite_interpreter_quant.get_output_details()

    tflite_interpreter_quant.resize_tensor_input(
        input_details[0]['index'], X_test.shape)
    tflite_interpreter_quant.resize_tensor_input(
        output_details[0]['index'], y_test.shape)
    tflite_interpreter_quant.allocate_tensors()

    input_details = tflite_interpreter_quant.get_input_details()
    output_details = tflite_interpreter_quant.get_output_details()

    # Run inference
    tflite_interpreter_quant.set_tensor(input_details[0]['index'], X_test)

    tflite_interpreter_quant.invoke()

    tflite_q_model_predictions = tflite_interpreter_quant.get_tensor(
        output_details[0]['index'])
    #print("\nPrediction results shape:", tflite_q_model_predictions.shape)
    # Convert prediction results to Pandas dataframe, for better visualization

    tflite_q_pred_dataframe = pd.DataFrame(tflite_q_model_predictions)
    tflite_q_pred_dataframe.columns = actions

    # print("Quantized TFLite model prediction results")
    # print(tflite_q_pred_dataframe)

    tf_model_predictions = sign_model(X_test)
    #print("Prediction results shape:", tf_model_predictions.shape)

    tf_pred_dataframe = pd.DataFrame(tf_model_predictions.numpy())
    tf_pred_dataframe.columns = actions

    text_file = open("./templates/testdata.html", "w")
    text_file.write("<h2>TF model prediction results: </h2>")

    text_file = open("./templates/testdata.html", "a")
    text_file.write(tf_pred_dataframe.to_html())

    text_file.write("<h2>TFLite model prediction results: </h2>")
    text_file.write(tflite_pred_dataframe.to_html())

    text_file.write("<h2>Quantized TFLite model prediction results: </h2>")
    text_file.write(tflite_q_pred_dataframe.to_html())

    # Concatenate results from all models

    all_models_dataframe = pd.concat([tf_pred_dataframe,
                                      tflite_pred_dataframe,
                                      tflite_q_pred_dataframe],
                                     keys=['TF Model', 'TFLite',
                                           'Quantized TFLite'],
                                     axis='columns')
    all_models_dataframe = all_models_dataframe.swaplevel(
        axis='columns')[tflite_pred_dataframe.columns]

    def highlight_diff(data, color='yellow'):
        attr = 'background-color: {}'.format(color)
        other = data.xs('TF Model', axis='columns', level=-1)
        return pd.DataFrame(np.where(data.ne(other, level=0), attr, ''),
                            index=data.index, columns=data.columns)

    final_df = all_models_dataframe.style.apply(highlight_diff, axis=None)

    text_file.write("<h2>Comparison between models: </h2>")
    text_file.write(final_df.to_html())

    # Concatenation of argmax and max value for each row
    def max_values_only(data):
        argmax_col = np.argmax(data, axis=1).reshape(-1, 1)
        max_col = np.max(data, axis=1).reshape(-1, 1)
        return np.concatenate([argmax_col, max_col], axis=1)

    # Build simplified prediction tables
    tf_model_pred_simplified = max_values_only(tf_model_predictions)
    tflite_model_pred_simplified = max_values_only(tflite_model_predictions)
    tflite_q_model_pred_simplified = max_values_only(
        tflite_q_model_predictions)

    # Build DataFrames and present example
    columns_names = ["Label_id", "Confidence"]
    tf_model_simple_dataframe = pd.DataFrame(tf_model_pred_simplified)
    tf_model_simple_dataframe.columns = columns_names
    tf_confidence = np.mean(tf_model_simple_dataframe["Confidence"])

    tflite_model_simple_dataframe = pd.DataFrame(tflite_model_pred_simplified)
    tflite_model_simple_dataframe.columns = columns_names
    tflite_confidence = np.mean(tflite_model_simple_dataframe["Confidence"])

    tflite_q_model_simple_dataframe = pd.DataFrame(
        tflite_q_model_pred_simplified)
    tflite_q_model_simple_dataframe.columns = columns_names
    tflite_q_confidence = np.mean(
        tflite_q_model_simple_dataframe["Confidence"])

    # print("Confidence in TF Model: ", tf_confidence)
    # print(tf_model_simple_dataframe)

    # print("Confidence in TF Lite Model: ", tflite_confidence)
    # print(tflite_model_simple_dataframe)

    # print("Confidence in TF Lite Quant Model: ", tflite_q_confidence)
    # print(tflite_q_model_simple_dataframe)

    text_file.write("<h2>Confidence in TF Model: </h2>")
    text_file.write(tf_model_simple_dataframe.to_html())

    text_file.write("<h2>Confidence in TF Lite Model: </h2>")
    text_file.write(tflite_model_simple_dataframe.to_html())

    text_file.write("<h2>Confidence in Quantized TF Lite Model: </h2>")
    text_file.write(tflite_q_model_simple_dataframe.to_html())

    text_file.close()
