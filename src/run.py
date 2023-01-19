from flask import Flask, request, jsonify
import pickle
import cv2
import numpy as np
from os import listdir

app = Flask(__name__)
app.debug = True
model = pickle.load(open("models/dide1.pkl", 'rb'))

label = list(listdir("data/PlantVillage"))

@app.route("/api/classify/", methods=["POST"])
def classify():
    print("test0")
    img = request.files['image']
    print("TEST1")
    img.save('temp.jpg')
    print("TEST2")
    image = cv2.imread('temp.jpg')
    resized = cv2.resize(image, (255, 255))
    print("TEST3")
    prediction = model.predict(np.array([resized]))
    print("TEST4")
    return jsonify({'prediction': label[np.argmax(prediction)]})



@app.route("/")
def index_test():
    return "test"
