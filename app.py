from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random
from database import get_db
from models import ChatHistory
from sqlalchemy.orm import Session

# Initialize Flask app
app = Flask(__name__)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

# Load trained model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# --- Helper functions ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_data = bow(sentence, words)
    res = model.predict(np.array([bow_data]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if len(ints) == 0:
        return "I'm not sure I understand. Can you rephrase?"
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand."

# --- Flask Routes ---
@app.route("/")
def home():
    return render_template("index.html")
    return "✅ Chatbot Flask API is running!"

@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        # Predict bot response
        ints = predict_class(user_message)
        bot_response = get_response(ints, intents)

        # Save chat history to DB
        db: Session = next(get_db())
        new_chat = ChatHistory(user_message=user_message, bot_response=bot_response)
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)

        return jsonify({"response": bot_response})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
