from flask import Flask, render_template, request, jsonify  # flask==1.1.1
from IntentClassifier import IntentClassifier
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

IP_ADDR = "192.168.0.102"
PORT = 8000
MODEL_PATH = os.path.dirname(__file__)+'/model/MyBot.h5'
TOKENIZER_PATH = os.path.dirname(__file__)+'/model/MyBot_tokenizer.json'

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/bot/", methods=['POST'])
def bot():
    text = request.json['text']
    res = classifier.getResponse(text)
    return jsonify({"response":res})
    
if __name__ == "__main__":
    responses = ['The temperature is 30 degrees today', 'The temperature is expected to be around 40 degrees tomorrow', 'The time is 10 A.M']
    
    classifier = IntentClassifier(5000, responses)
    classifier.setMaxLen(10)
    classifier.loadModel(MODEL_PATH, TOKENIZER_PATH)

    app.run(host=IP_ADDR, port=PORT, debug=False)
