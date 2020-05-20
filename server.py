from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)


model = joblib.load(open('models/pipe_clf_checkpoint.joblib', 'rb'))
model_classification = model['pipeline_clf']

@app.route('/')
def hello_world():
    return 'Sentiment analysis API!'

@app.route('/upload')
def upload_file():
    return 'TODO'

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentiment_pred = model_classification.predict(data["input"])
    output_text = "Text: " + str(data["input"])
    output = "class: " + str(sentiment_pred)
    return jsonify(output_text, output)

if __name__ == '__main__':
    app.run(host="localhost", port=3000, debug=True)
