from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np #numpy==1.18.5
import tensorflow #tensorflow==2.2.0
import json #json==2.0.9
import random

class CustomCallback(tensorflow.keras.callbacks.Callback):

  def on_epoch_end(self, epochs, logs=None):
    if logs.get('accuracy')>=0.95 and logs.get('loss')<=0.05:
      self.model.stop_training=True


class IntentClassifier():

  def __init__(self):
    '''  
    Arguments:
    ----------
    tokenizer       --  tensorflow-keras Tokenizer object to convert words to indices
    embed_dim       --  integer representing size of the word vector
    predThresh      --  threshold for allowing high confidence predictions
    intentResponses --  python dictionary containing integers corresponding to intents as keys and 
                        a python list containing responses from that list
                        E.g: intentResponses = {0: ['Hey', 'Good Morning']}
    '''
    self.tokenizer = Tokenizer()
    self.embed_dim = 300 
    self.predThresh = 0.9
    self.intentResponses = dict()


  def initialize(self):
    """  
    Description:
    ------------
    This function prepares the data in the format required for training using tensorflow-keras
    """
    examples = []
    labels = []
    
    """ 
    Generate 'examples' which is a python list of all the training sentences for all the intents combined and
    'labels' which is a python list of integers representing the intent to which every training example in 'examples' belongs
    E.g:
    examples        = ['abc', 'def', 'gef']
    labels          = [ 0, 1, 2]  
    intentResponses = {0: ['Hey', 'Good Morning']}
    """
    c = 0
    for intent, val in self.dataset.items():
        for example in val["examples"]:
            examples.append(example)
            labels.append(c)
        self.intentResponses[c] = val["responses"]
        c+=1

    numIntents = c

    # Convert labels into one_hot vectors
    self.labels = to_categorical(labels)

    """ 
    Once fit, the Tokenizer provides 4 attributes that you can use to query what has been learned about your documents:
    - word_counts:      A dictionary of words and their counts.
    - word_docs:        A dictionary of words and how many documents each appeared in.
    - word_index:       A dictionary of words and their uniquely assigned integers.
    - document_count:   An integer count of the total number of documents that were used to fit the Tokenizer.
    """
    self.tokenizer.fit_on_texts(examples)
    self.vocab_size = len(self.tokenizer.word_index)+1000
    # Convert each sentence into sequence of indices
    seq = self.tokenizer.texts_to_sequences(examples)
    
    # Find length of the maximum length sentence and pad other sentences with zeros uptill this maximum length
    maxLen = 0
    for s in seq:
        seqLen = len(s)
        if seqLen>maxLen:
            maxLen  = seqLen
    
    self.max_len = maxLen
    self.trainData = pad_sequences(seq, maxlen=self.max_len, padding='post')

    self.model = Sequential([
        Embedding(self.vocab_size, self.embed_dim, input_length = self.max_len),
        LSTM(256),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(numIntents, activation='softmax')
      ])
    self.model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        

  def saveModel(self, modelName):
    self.model.save(modelName+".h5")
    with open(modelName+'_tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(self.tokenizer.to_json(), ensure_ascii=False))


  def loadModel(self, modelPath, tokenizerPath):
    with open(tokenizerPath) as f:
      data = json.load(f)
      self.tokenizer = tokenizer_from_json(data)
    self.model = load_model(modelPath)
    self.model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


  def train(self, dataset, epochs, retrain=False):
    '''  
    Arguments:
    ----------
    dataset --  python dictionary containing intents as keys and their responses and examples as values
                @format: dataset = {"intent-name": {"examples":[], "responses":[]}}
    epochs  --  no. of training iterations
    retrain --  boolean value representing whether to train a new model or an existing model which is 
                loaded using the loadModel() function
    '''
    self.dataset = dataset
    if not retrain:
      self.initialize()
    callback = CustomCallback()

    self.model.fit(self.trainData, self.labels, epochs=epochs, callbacks=[callback])


  def predict(self, text):
    """  
    Description:
    ------------
    This function classifies the text given as input and identifies the intent. 
    It then returns a random response associated with the identified intent.
    """
    # Convert text into integers using the word to index mapping 
    seq = self.tokenizer.texts_to_sequences([text])
    # Pad the sequences to make all sequences of equal length
    data = pad_sequences(seq, maxlen=self.max_len, padding='post')
    prediction = self.model.predict([data])
    confidence = np.max(prediction)

    if confidence>=self.predThresh:
        responses = self.intentResponses[np.argmax(prediction)]
        response = random.choice(responses)
    else:
        response = "I am sorry, I don't understand it."

    return response


