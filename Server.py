import logging, os #logging==0.5.1.2
from IntentClassifier import IntentClassifier
from flask import Flask, render_template, request, jsonify  # flask==1.1.1

""" Disable all warnings """
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IP_ADDR = "localhost"
PORT = 8000
IMG_SIZE = 28
NUM_EPOCHS_TO_TRAIN = 1000

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=['POST'])
def train():
    dataset = request.json # dataset = {"intent-name": {"examples":[], "responses":[]}}
    
    try:
        classifier.train(dataset, NUM_EPOCHS_TO_TRAIN)
        res = "successful"
    except:
        res = "failed"
    
    return jsonify(response=res)

@app.route("/predict", methods=['POST'])
def predict():
    text = request.json["text"]
    res = classifier.predict(text)
    
    return jsonify(response=res)

if __name__ == "__main__":
    classifier = IntentClassifier()
    app.run(host=IP_ADDR, port=PORT, debug=True)