from flask import Flask, request, jsonify
import pickle
import cv2
import numpy as np

app = Flask(__name__)
app.debug = True
model = pickle.load(open("models/first_cnn.pkl", 'rb'))

label = ['bacteria', 'brown', 'smut']

@app.route("/api/classify/", methods=["POST"])
def classify():
    img = request.files['image']
    img.save('temp.jpg')
    image = cv2.imread('temp.jpg')
    resized = cv2.resize(image, (180, 180))
    prediction = model.predict(np.array([resized]))
    return jsonify({'prediction': label[np.argmax(prediction)]})



@app.route("/")
def index_test():
    return "test"
