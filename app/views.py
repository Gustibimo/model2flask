from flask import Flask, request, jsonify, render_template, redirect, make_response
import numpy as np
import joblib
import os
from app import api

model = joblib.load(open('models/pipe_clf_checkpoint.joblib', 'rb'))
model_classification = model['pipeline_clf']

@api.route('/')
def hello_world():
    return 'Sentiment analysis API!'

api.config["IMAGE_UPLOADS"] = "/home/safia/workspaces/labs/datasci/model2flask/app/static/uploads"
#TODO: upload test data to determine class output
@api.route('/upload', methods= ['GET', 'POST'])
def upload_file():
    if request.method == "POST":
        if request.files:
            print(request.cookies)
            image = request.files["image"]

            if image.filename == "":
                print("Image must have filename")
                return redirect(request.url)
            image.save(os.path.join(api.config["IMAGE_UPLOADS"], image.filename))

            print(image)
            return redirect(request.url)

    return render_template('upload.html')

@api.route("/uploader", methods=["GET", "POST"])
def upload_model():
    if request.method == "POST":
        filesize = request.cookies.get("filesize")
        file = request.files["file"]

        print(f"filesize: {filesize}")
        print(file)

        res = make_response(jsonify({"message": f"{file.filename} uploaded"}), 200)

        return res
    return render_template("/upload_model.html")

@api.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentiment_pred = model_classification.predict(data["input"])
    output_text = "Text: " + str(data["input"])
    output = "Prediction: " + str(sentiment_pred)
    return jsonify(output_text, output)