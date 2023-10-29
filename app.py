import os
from flask import Flask, render_template, redirect, request, url_for, send_file, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
# import tensorflow as tf
from tensorflow.keras.models import load_model
app = Flask(__name__)
load = ('./model/CPNcolabfulldata94.h5')
model = load_model(load)
img_size = 50

# Load the chest x-ray classification model
xray_model = load_model('./model/model2.h5')
threshold_prob = 0.5

@app.route('/')
def index():
    return render_template('index.html')

# //RestfullAPI
@app.route('/perventions')
def prevetnions():
    return render_template('preventions.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/Home')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_name = request.files['image'].filename
    upload_dir = './Uploaded_images'
    image_path = os.path.join(upload_dir, image_name)
    request.files['image'].save(image_path)
    
    # Check if the uploaded image is a chest x-ray image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (50, 50))
    image = image.astype('float32')
    image /= 255
    image = np.expand_dims(image, axis=0)
    is_chest_xray = xray_model.predict(image)
    if is_chest_xray[0][0] < threshold_prob:
        return jsonify(error="Invalid image. Please upload a chest x-ray image.")
    
    # Preprocess the chest x-ray image and make a prediction
    predictions = model.predict(image)[0]
    class_names = ['NORMAL', 'PNEUMONIA', 'COVID-19']
    prediction = class_names[np.argmax(predictions)]
    max_prob = round(max(predictions) * 100,2)
    max_prob_formatted = "{:.0%}".format(max_prob / 100)

    return jsonify(max_prob=max_prob_formatted, prediction=prediction)



if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)



























# @app.route('/predict', methods=['POST'])
# def predict():

#     image_name = request.files['image'].filename
#     upload_dir = './Uploaded_images'
#     image_path = os.path.join(upload_dir, image_name)
#     request.files['image'].save(image_path)
#     image = request.files['image'].read()
#     image = np.asarray(bytearray(image))
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (50, 50))
#     image = image.astype('float32')
#     image /= 255
#     image = np.expand_dims(image, axis=0)

#     class_names = ['NORMAL', 'PNEUMONIA', 'COVID-19']
#     predictions = model.predict(image)[0]
#     predictions = predictions.tolist()
    

#     prediction = class_names[np.argmax(predictions)]

#     max_prob = round(max(predictions) * 100,2)

#     max_prob_formatted = "{:.0%}".format(max_prob / 100)

#     # we can also return the jsonify object

#     return jsonify(max_prob=max_prob_formatted, prediction=prediction)

# return render_template("sub.html", prediction=prediction, max_prob=max_prob,)
# for the probability of three classes
# class_names = ['Normal', 'Pneumonia', 'COVID']
# result = {}
# for i in range(len(predictions)):
#     result[class_names[i]] = predictions[i]
# return jsonify(result)
# max_prob=round(max(prediction)*100, 2)


