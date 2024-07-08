import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from googletrans import Translator
from flask import Flask, render_template, request, jsonify, session
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
import time
langVariable = "0"

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    print("This is the response",res)
    return res

def translate_to_devanagari(text):
    translator = Translator()
    translation = translator.translate(text, src='en', dest='hi')
    return translation.text

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='hi', dest='en')
    return translation.text


app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = 'your_secret_key'
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/showmap")
def map():
    return render_template("test.html")

@app.route("/get")
def get_bot_response():
    langVariable = session.get('langVariable', "0")
    print("This should be the lang variable",langVariable)
    print(type(langVariable))
    start_time = time.time()
    if (langVariable == "1" ):
        userText = request.args.get('msg')
        print(userText)
        proText = translate_to_english(userText)
        print(proText)

        init_res =  chatbot_response(proText)

        print("This is the",init_res)
        print(type(init_res))
        if isinstance(init_res,dict):
            init_res['res'] = translate_to_devanagari(init_res['res'])
        else:
            init_res = translate_to_devanagari(init_res)
        end_time = time.time()
        print("Delay is : ",end_time-start_time)
        return init_res
    else:
        print("Simon")
        userText = request.args.get('msg')
        text = chatbot_response(userText)
        end_time = time.time()
        print("Delay is : ",end_time-start_time)
        return text

@app.route('/your_flask_route', methods=['POST'])
def receive_variable():
    data = request.get_json()
    langVariable = data.get('variable')
    session['langVariable'] = langVariable
    # Process the variable as needed
    # For example, print it and send a response back to the client
    print('Received variable from JavaScript:', langVariable)
    # You can send a response back to the client if needed
    return jsonify({'message': 'Variable received successfully'})

if __name__ == '__main__':
    app.run(debug=True)



if __name__ == "__main__":
    app.run()