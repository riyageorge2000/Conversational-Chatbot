import streamlit as st
import random
import json
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')

# Initialize the Lancaster Stemmer
stemmer = LancasterStemmer()

# Load the pre-trained model and other necessary files
model = load_model('model_keras.h5')
intents = json.loads(open('intents.json').read())
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']

# Tokenize and preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Return a bag of words from the user input
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("Found in bag: %s" % w)
    return(np.array(bag))

# Predict the intent of the user input
def predict_class(sentence, model, words, classes):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def classify(sentence):
    # Prediction or To Get the Possibility or Probability from the Model
    results = model.predict(np.array([bow(sentence, words)]))[0]
    # Exclude those results which are Below Threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # Sorting is Done because higher Confidence Answer comes first.
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))  # Tuple -> Intent and Probability
    return return_list

# Get a response from the chatbot
def response(sentence, userid='123', show_details=True):
    results = classify(sentence)
    if results:
        for i in intents['intents']:
            if i['tag'] == results[0][0]:
                return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

# Streamlit App
st.title("ChatFlow: A Conversational Chatbot")

# List to store the conversation
conversation = st.session_state.get('conversation', [])

# User Input
user_input = st.text_input("You:")

# Bot Response
if user_input.lower() != 'quit':
    bot_response = response(user_input)
    conversation.append({"role": "user", "text": user_input})
    conversation.append({"role": "bot", "text": bot_response})
    st.session_state.conversation = conversation

# Display previous conversations
for entry in conversation:
    if entry["role"] == "user":
        st.text("You: " + entry["text"])
    elif entry["role"] == "bot":
        st.text("Bot: " + entry["text"])
