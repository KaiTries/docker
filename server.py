import requests
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import ImageHandler
import numpy as np


url = 'http://localhost:8501/v1/models/img_classifier:predict'

api = Flask(__name__)
CORS(api, support_credentials=True)

@api.route("/")
def index():
    return "currently running"



@api.route("/mnist",methods=["GET","POST"])
def get_mnist():
    encoded_picture = request.json[22:]

    decodedImg = ImageHandler.retrieveB64(encoded_picture)
    readyImage = ImageHandler.ImageForModel(decodedImg)
    final = ImageHandler.rec_digit(readyImage)

    data = json.dumps({"signature_name":"serving_default","instances":
                       final.tolist()})

    headers = {"content-type":"application/json"}

    json_response = requests.post(url,data=data,headers=headers)
    predictions = json.loads(json_response.text)['predictions']

    predictions_single = np.argmax((predictions[0]))


    response = jsonify({"Prediction":str(predictions_single)})

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


if __name__ == "__main__":
    api.run(debug=True)

