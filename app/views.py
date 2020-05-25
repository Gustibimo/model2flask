from flask import Flask, request, jsonify, render_template, redirect, make_response
import numpy as np
import joblib
import os

from flask_restful import reqparse

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
            image = request.files["image"]

            if image.filename == "":
                print("Image must have filename")
                return redirect(request.url)
            image.save(os.path.join(api.config["IMAGE_UPLOADS"], image.filename))

            return redirect(request.url)

    return render_template('upload.html')

api.config["MODEL_UPLOADS"] = "/home/safia/workspaces/labs/datasci/model2flask/models"

@api.route("/uploader", methods=["GET", "POST"])
def upload_model():
    if request.method == "POST":
        filesize = request.cookies.get("filesize")
        file = request.files["file"]
        file.save(os.path.join(api.config["MODEL_UPLOADS"], file.filename))

        print(f"filesize: {filesize}")
        res = make_response(jsonify({"message": f"{file.filename} uploaded"}), 200)

        return res
    return render_template("/upload_model.html")

@api.route('/api/v1/sentiment', methods=['GET','POST'])
def prediction():

    if request.method == 'POST':
        data = request.get_json(force=True)
        sentiment_pred = model_classification.predict(data["input"])
        output_text = "Text: " + str(data["input"])
        output = "Prediction: " + str(sentiment_pred)
        label= list()
        for s in sentiment_pred:
            if s == 0:
                label.append("Negative")
            else:
                label.append("Positive")
        output_label = "Label: " + str(label)
        return jsonify(output_text, output, output_label)
    if request.method == 'GET':
        args = list()
        args.append(request.args.get("text"))
        sen = model_classification.predict(args)
        get_label = list()
        for s in sen:
            if s == 0:
                get_label.append("Negative")
            else:
                get_label.append("Positive")
        output_labels = "Label: " + str(get_label)

        return jsonify(str(output_labels))