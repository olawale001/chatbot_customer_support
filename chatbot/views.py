import json
import nltk
import numpy as np
import random
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from django.http import JsonResponse

lemmatizer = WordNetLemmatizer()

model = tf.keras.models.load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json', 'r') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)    
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)            

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intent_list, intent_json):
    tag = intent_list[0]["intent"]
    for i in intent_json["intents"]:
        if i['tag'] == tag:
            return random.choice(i['responses'])
        

# def chatbot_response(request):
#     if request.method == 'POST':
#         user_message = request.POST.get('message', '')
#         predicted_intents = predict_class(user_message)
#         response = get_response(predicted_intents, intents)
#         return JsonResponse({"response": response})
#     return JsonResponse({"error": "Invalid request"}, status=400)

def chatbot_response(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '').strip()
        if not user_message:
            return JsonResponse({"error": "Message cannot be empty"}, status=400)
        
        try:
            predicted_intents = predict_class(user_message)
            response = get_response(predicted_intents, intents)
            return JsonResponse({"response": response})
        except FileNotFoundError as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request"}, status=400)
