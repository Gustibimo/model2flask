from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)


model = joblib.load(open('models/pipe_clf_checkpoint.joblib', 'rb'))
model_classification = model['pipeline_clf']

@app.route('/')
def hello_world():
    return 'Sentiment analysis API!'

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sentiment_pred = model_classification.predict(data["input"])
    output_text = "Text: " + str(data["input"])
    output = "class: " + str(sentiment_pred)
    return jsonify(output_text, output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
