from flask import Flask, request, jsonify
import numpy as np
import joblib

api = Flask(__name__)


model = joblib.load(open('models/pipe_clf_checkpoint.joblib', 'rb'))
model_classification = model['pipeline_clf']

@api.route('/')
def hello_world():
    return 'Sentiment analysis API!'

@api.route('/upload')
def upload_file():
    return 'TODO'

@api.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentiment_pred = model_classification.predict(data["input"])
    output_text = "Text: " + str(data["input"])
    output = "Prediction: " + str(sentiment_pred)
    return jsonify(output_text, output)

if __name__ == '__main__':
    api.run(host="localhost", port=9000, debug=True)
