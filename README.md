# model2flask

mMchine learning model deployment using Flask

# Available model

- SVM Classification Model for Simple Sentiment Analysis
- Convolutional Neural Network for image captioning

# Available API

- `api/v1/sentiment`
- `api/v1/imgcaption`
- `/uploader`

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
> curl -X POST "http://localhost:9000/api/sentiment" -H "Content-Type: application/json" -d @test_data.json

```

or you use **GET** method

```
> curl -X GET "http://localhost:9000/apiv1/sentiment?text=Python is awesome"
```



- sample data for sentiment analysis

```
{
    "input" : ["I hate wageningen", "I love utrecht"]
}
```

- ouput

```
> {

  "Text: ['I love wageningen', 'I love utrecht']", 
  "Prediction: [1 1]", 
  "Label: ['Positive', 'Positive']"

}
```




