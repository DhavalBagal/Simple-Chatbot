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

class CustomCallback(tensorflow.keras.callbacks.Callback):

  def on_epoch_end(self, epochs, logs=None):
    if logs.get('accuracy')>=0.98 and logs.get('loss')<=0.01:
      self.model.stop_training=True


class IntentClassifier():

  def __init__(self, vocab_size, responses):
    '''  
    Arguments:
    ----------
        vocab_size => Integer denoting the size of the dictionary
        responses => Python list consisting of responses to be sent upon determining the intent
    '''
    self.tokenizer = Tokenizer()

    # Dimension of the word embedding used to represent the words
    self.embed_dim = 300

    self.responses = responses
    self.vocab_size = vocab_size
    self.predThresh = 0.9

  
  def getResponse(self,text):

    # Convert text into integers using the word to index mapping 
    seq = self.tokenizer.texts_to_sequences([text])

    # Pad the sequences to make all sequences of equal length
    data = pad_sequences(seq, maxlen=self.max_len, padding='post')

    data = np.array(data)
    prediction = self.model.predict([data])

    return self.responses[np.argmax(prediction)]


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


  def initializeModel(self):
    self.model = Sequential([
        Embedding(self.vocab_size, self.embed_dim, input_length = self.max_len),
        LSTM(256),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(3,activation='softmax')
      ])

    self.model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])


  def trainModel(self, epochs, retrain=False):
    '''  
    Arguments:
    ----------
        data => Python list containing fixed sized sublists of integers representing words of the sentences
        labels => One hot vectored labels
        retrain => Boolean value representing whether to train a new model or an existing model which is 
                   loaded using the loadModel() function
    '''

    if not retrain:
      self.initializeModel()

    callback = CustomCallback()
    self.model.fit(self.data, self.labels, epochs=epochs, callbacks=[callback])


  def setMaxLen(self, max_len):
    self.max_len = max_len


  def prepareData(self, dataset, labels):
    ''' 
    Arguments:
    ----------
        dataset => Python list of statements for training the chatbot
        labels => Python list of integers starting from 0 which are mapped to the intents
        responses => Python list consisting of responses to be sent upon determining the intent

    Format:
    -------
        dataset: ['abc', 'xyz','bvc', 'mln', 'gef']
        labels: [0,0,1,1,1]
        responses: ['def','pqr']
    '''

    # Convert labels into one_hot vectors
    self.labels = to_categorical(labels)

    # Map words to indices
    self.tokenizer.fit_on_texts(dataset)

    # Convert sentences into sequence of indices
    seq = self.tokenizer.texts_to_sequences(dataset)

    # Find length of the maximum length sentence and pad other sentences with zeros uptill this maximum length
    self.data = pad_sequences(seq, maxlen=self.max_len, padding='post')


    