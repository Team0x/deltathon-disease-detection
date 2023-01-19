from flask import Flask, request

app = Flask(__name__)

@app.route("/api/classify/", methods=["POST", "GET"])
def classify():
    return request.url


@app.route("/")
def index_test():
    return "Test"