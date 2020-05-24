# model2flask

Machine learning model deployment using Flask

# Available model

- SVM Classification Model for Sentiment Analysis
- Convolutional Neural Network for image captioning

# Prerequisite

- Python 3.7
- Flask
- Numpy
- Sklearn
- Pickle

# Setup

1. Install library depedencies in requirements.txt

```
> pip install -r requirements.txt
```

2. Install tensorFlow
```
> sudo apt-get install tensorflow
```

# Usages

- To run classification using inputed json data

```
> curl -X POST "http://localhost:9000/api" -H "Content-Type: application/json" -d @test_data.json

```

- sample data for sentiment analysis

```
{
    "input" : ["I hate wageningen", "I love utrecht"]
}
```




