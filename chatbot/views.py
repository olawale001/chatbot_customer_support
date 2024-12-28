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

def get_response(intent_list, intents_json):
    if not intent_list:
        for i in intents_json['intents']:    
            if i['tag'] == 'fallback':
                return random.choice(i['response'])

    tag = intent_list[0]['intent']            
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['response'])

def get_response_with_context(intent_list, intent_json, session_context):
    tag = intent_list[0]["intent"]
    context_set = None

    for i in intent_json["intents"]:
        if i['tag'] == tag:
            context_set = i.get('context_set', None)
            if not i.get('context_filter') or session_context == i.get('context_filter'):
               return random.choice(i['responses']), context_set

    return "I am not sure how to response to that", context_set           
        

def chatbot_response(request):
    if request.method == 'POST':
        user_message = request.POST.get('message', '')
        session_context = request.session.get('context', None)
        user_feedback = request.POST.get('feedback', None)

        if user_feedback:
            with open('feedback.txt', 'a') as file:
                file.write(f"{user_message}: {user_feedback}\n")

            return JsonResponse({"response": "Thanks for your feedback"})    


        predicted_intents = predict_class(user_message)
        response, new_context = get_response_with_context(predicted_intents, intents, session_context)

        if new_context:
            request.session['context'] = new_context

        return JsonResponse({"response": response})
    return JsonResponse({"error": "Invalid request"}, status=400)