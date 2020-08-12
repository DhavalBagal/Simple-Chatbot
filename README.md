# Simple-Chatbot

The project allows you to train your personal chatbot on your own data. You can input the intents and the examples either through the web application or by uploading a JSON file for your dataset. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing purposes.

### Prerequisites

You need to have the following packages installed on your device before running the project.

	python==3.6 (or above)
	numpy==1.18.5
	tensorflow==2.2.0
	json==2.0.9
	flask==1.1.1
	
### Directory Structure and Breakdown

**`Server.py`** - Server side script that interacts with the web application

**`IntentClassifier.py`** - RNN Model in Tensorflow-Keras for intent classification

**`data.json`** - Sample JSON file for the dataset 

**`templates/index.html`** - Web application

**`static/`** - JS classes, font file and images

### Running the project

1. Clone the repository to your local system.
2. Change your default directory to the directory where you cloned the project using ***cd*** command.
3. Open your terminal and execute the Server. py script
		
		python3 Server.py

4. Go to your browser and enter the following:
	
		http://localhost:8000/
	
## Built with

**React** - Framework for Web application

**Flask** - Server Side Backend

**Tensorflow-Keras** - Building the RNN model
