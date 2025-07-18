from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow as tf
import wikipedia
from duckduckgo_search import DDGS

# Initialize app and stemmer
app = Flask(__name__)
stemmer = LancasterStemmer()
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

# Preprocess user sentence
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

# Classify intent
ERROR_THRESHOLD = 0.25
def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

# Fallback: search web
def fallback_web_answer(query):
    try:
        # Try DuckDuckGo first
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                if r.get("body"):
                    return r["body"]
        # Fallback to Wikipedia
        return wikipedia.summary(query, sentences=2)
    except Exception:
        return "I'm not sure how to answer that yet. Try rephrasing!"

# Get chatbot response
def get_response(sentence):
    results = classify(sentence)
    if results and results[0][1] >= 0.5:
        tag = results[0][0]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    else:
        return fallback_web_answer(sentence)

# API routes
@app.route("/", methods=["GET"])
def home():
    return "✅ Chatbot API (with Web Search) is running!"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
