from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow as tf

# Initialize app and stemmer
app = Flask(__name__)
stemmer = LancasterStemmer()

# Download NLTK tokenizer
nltk.download('punkt')

# Load intents and training data
with open("intents.json") as file:
    intents = json.load(file)

data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']

# Load model
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(words)])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(classes), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load("model.tflearn")

# Preprocess and predict
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

ERROR_THRESHOLD = 0.25

def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

def get_response(sentence):
    results = classify(sentence)
    if results:
        for intent in intents["intents"]:
            if intent["tag"] == results[0][0]:
                return random.choice(intent["responses"])
    return "I'm not sure I understand. Can you rephrase?"

# Define routes
@app.route("/", methods=["GET"])
def home():
    return "✅ Chatbot API is running!"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
